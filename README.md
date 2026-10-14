# PRC Data Challenge 2026: taxi-out time prediction (team resilient-hedgehog)

Code and documentation of team **resilient-hedgehog** (MOCHITEC Inc.) for the
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) of the EUROCONTROL Performance Review
Commission and the OpenSky Network. Licence: GNU GPLv3 (`LICENSE`).

- **Final file:** §3 and [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md).
- **Results:** §10.
- **Documentation map:** §11.

## 1. Task and metric

- **Target.** `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` (seconds) for departures at 10 airports: EDDF, EDDM, EGLL,
  EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.
- **Training data.** The organisers' 2025 movements (`training_2025-MM-01_*.parquet`, all 12 months), with
  the NM flight record joined (`AOBT_3_flt`, `LOBT_flt`, …).
- **Leaderboard set.** January and July 2026 (`ranking.parquet`, 344,841 departures).
- **Final set.** January, February, June and July 2026 (`final_ranking.parquet` / `final_submitting.parquet`,
  670,790 departures). The final file is `resilient-hedgehog_final.parquet`.
- **Metric.** RMSE over all rows.

## 2. Model design

Map of the pipeline at v47: `docs/SOLUTION_OVERVIEW.md`. Short version:

1. **Two lanes, split on `AOBT_3_flt`.**
   - *Matched lane* (AOBT_3 present, about 98.5 % of rows).
   - *Unmatched lane* (no NM record). A separate LightGBM ("model B") is fitted. At LIRF its target is
     reparameterised on the schedule (`y − (MVT − SCHED)`).
2. **Anchored target.** On the matched lane the anchored members predict `y − proxy`, where `proxy = MVT − AOBT_3`
   (`src/prc/train.py`, `matched_target`). The plain-target member predicts `y`.
3. **5-member blend on matched rows**: `((a2o + pao)/2 + p2o + c2 + m4) / 4`.

   | member | model | production config |
   |---|---|---|
   | a2o | anchored LightGBM, tuned, gate retention (L6) and LIRF copy mixture (cmix) | `configs/main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml` |
   | pao | per-airport anchored LightGBM, L6 | `configs/main_flseq_osm_nolirf_l2_pa_opt_l6.yaml` |
   | p2o | plain-target LightGBM, L6 | `configs/main_flseq_osm_nolirf_plainy_l2_opt_l6.yaml` |
   | c2 | CatBoost | `configs/main_flseq_osm_nolirf_cb_l2.yaml` |
   | m4 | MLP, L6 | `configs/main_flseq_osm_nolirf_l2_mlp_l6.yaml` |

   Features: schedule and NM clocks, neighbouring departures and runway sequence (`src/prc/features.py`,
   `src/prc/features_extra.py`, `src/prc/levers.py`), stand/arrival occupancy, hourly METAR weather
   (`src/prc/weather.py`), and the OSM stand-to-runway distance (`src/prc/osm.py`).
   The unmatched lane is model B, blended at weight 0.25 with an "NM-blind" model outside LIRF. At LIRF it uses a
   gated copy mixture (`ucmix`) on rows with 0 < MVT − SCHED < 8 h.
4. **Learned ADS-B residual layer.** A LightGBM fitted on 2025 to `clip(y − b, ±3600)`, where `b` is the blend.
   Its inputs are ground-movement features from adsb.lol traces: push, first move, stand dwell, post-push stops and
   positions. These are joined to departures by callsign, or by position for rows without a callsign join. It
   corrects rows with ADS-B coverage only. Its correction is clipped to ±1800 s and is 0 at LTFM.
   - Components: `G12A`, `G7A` (LFPG), `GL4` (LIRF), `AIR`, `POS`, `RULE`.
   - Code: `scripts/adsb_*.py`, `analysis/local_adsb_*.py`, `analysis/local_posjoin_*.py`, and the final-set
     rebuild `docs/final_2026-10-11/adsb_layer.py` with its README `docs/final_2026-10-11/adsb_layer_README.md`.
5. **Appear tier.** Eight columns describing the aircraft's ADS-B position and timing relative to its own stand
   (`scripts/adsb_appear.py`), from OSM `parking_position` plus a 2026 ADS-B stand map. They are added to the
   residual model and applied as the increment `c_A12(b) − c_P12(b)`. Code: `docs/appear_tier_2026-10-05/`
   (`final_adsb.py` for the final set). In the final file the appear tier is in the January/July rows (from v48) and,
   as the increment, in 226,935 February/June rows. 2026-06-11 goes through the A12 layer instead, and only 78 appear
   rows exist for that day. See `docs/FINAL_SUBMISSION.md` §2.3 and §2.5.
6. **Overlays**, applied after the residual layer by `docs/final_2026-10-11/apply_overlays.py`:
   - lane-B deltas: NM-blind and ucmix;
   - LIRF matched schedule-copy overlay: `p' = p + 0.4·q·(MVT − SCHED − p)`, where q comes from a 2025 classifier;
   - LIRF unmatched copy band: 7 to 13.5 h late, predict `MVT − SCHED`;
   - LIRF unmatched time-zone class: per-prefix q blend;
   - **LOBT clip** last: project onto `[MVT − LOBT − 3606, MVT − LOBT + 3606]`. This window holds on every matched
     2025 row.

How each piece was validated: `docs/SOLUTION_OVERVIEW.md` §6 and the per-topic folders under `docs/`.

## 3. The final file and how it was produced

**The authoritative description is [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md).** Its §1 defines the
file's identity: source commit, md5, rows and validation result. This section is a summary.

- **The file.** `handoff/2026-10-13/final_v5_plus_febjun_appear.parquet`, md5 **`c0d4a1439862e68852b65f6854322a29`**,
  2,151,179 bytes, 670,790 rows. It was added in **`ea0e0a8`** (branch `claude/beautiful-davinci-ehugk4`, contained
  in `main`), deleted in `9849a14` and restored byte for byte in `0d43558`; it is in the tree on `main`.
  - **Its base** is `handoff/2026-10-13/final_from_v48.parquet` as committed in `25f6246` (md5 `be8d89dc…`, branch
    `claude/laughing-cray-vvwh0k`). The uploaded file = that base + the 12-month appear increment on 103,311 February
    and 123,624 June rows. January/July are unchanged (= v48).
  - Earlier versions of this README named the base `be8d89dc…` as the final file. It was not uploaded.
- **The upload.** Uploaded as `prc-2026-resilient-hedgehog/resilient-hedgehog_final.parquet` by `prc.submit --final`;
  the bucket object's `last_modified` is **2026-10-14 10:30:29Z** and its read-back md5 matched. The ledger lines
  (`"kind": "final"`) are at the end of `experiments/submissions.jsonl` (`8189b58`). It replaced an object of
  2026-10-09 (4,079,671 bytes, uploader unknown, whose scoring had failed).
  - **Scorer:** the result file written at 10:30:38Z says `"status": "Succeeded"`, `"used_pairs": 670790`, and
    contains no score (copy: `docs/final_phase_2026-10-11/final_result_2026-10-14.json`).
  - **Deadline:** the organisers' pages state "2026-10-14T12:00 CET (2026-10-14T12:00+02:00)" and
    "2026-10-14T10:00Z". The upload was at 10:30:29Z, after 10:00Z. Whether it is accepted is for the organisers.
- **Why this file** (`docs/FINAL_SUBMISSION.md` §4). On 2026-10-14 the Jan/Jul part of the laughing-cray v6 file
  scored 234.5741 on the leaderboard (v50), while v48 resubmitted unchanged scored 234.4099 (v51), exactly v48's
  score. So the Jan/Jul rows stay v48, and February/June get the appear increment so that they follow v48's recipe.

**How the rows were produced** (`docs/FINAL_SUBMISSION.md` §2):

1. **January and July rows: v48, unchanged.**
   - The file is `handoff/2026-10-08/C15_appear12_cov26eddf.parquet`, md5 `8ee6679f…`, public leaderboard 234.4099.
     All 344,841 rows are equal.
   - v48 = C15 (v47) + the 12-month appear delta + the EDDF coverage-matched delta (below).
   - v48 itself is the end of a chain of per-submission delta layers, v33 → v48. See `docs/SOLUTION_OVERVIEW.md` §7,
     `docs/batch_2026-10-06/README.md`, `handoff/2026-10-08/ON_C15.md` and
     `docs/final_2026-10-11/adsb_layer_README.md` §1.
2. **February and June rows: the chain**, run on the final data (recipe: `handoff/2026-10-13/README.md` as of
   `25f6246`, translated in `docs/FINAL_SUBMISSION.md` §2.2 and §5.3).
   1. **Members**, trained with `python -m prc.pipeline --config <member config> --skip-cv` on the final raw data.
      The code is peaceful-babbage `6a97bf0`.
      - The members are a2o, pao, p2o and m4 with 3 seeds each, and c2 (CatBoost) with 1 seed. The configs are in the
        §2 table.
      - The `fl_*` / `seq_*` columns of matched rows were rebuilt on the 2025 + 2026 union
        (`analysis/local_audit_fix2026.py`).
      - The predictions are committed in `handoff/2026-10-13/members/`.
   2. **Blend.** Matched rows: `((a2o + pao)/2 + p2o + c2 + m4) / 4`. Unmatched rows: model B. The result is
      `handoff/2026-10-13/pre_adsb_base_blend5_final.parquet`.
   3. **ADS-B residual layer.**
      - Our rebuilt layer (`docs/final_2026-10-11/adsb_layer.py apply`) covers all 28 February days and 29 of the 30
        June days.
      - 2026-06-11 had no adsb.lol data in that ingest, so it uses the A12 layer
        (`docs/appear_tier_2026-10-05/final_adsb.py`). Its effect there is close to zero
        (`docs/FINAL_SUBMISSION.md` §2.3).
      - This layer has no appear term; step 6 adds it.
   4. **Overlays and LOBT clip:** `docs/final_2026-10-11/apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`.
      See `docs/final_2026-10-11/overlays_README.md` and `laneB_deltas.py`.
   5. Floor at 0, round, int32. This gives the base `be8d89dc…`.
   6. **Appear increment.** `handoff/2026-10-13/feb_jun_appear_increment_on_blend5.parquet` (md5 `fe3aedc0…`): the
      12-month appear term `c_A12 − c_P12` (`c_A12thin − c_P12` at EDDF), from `final_adsb.py --increment` on the
      blend. It is added on 103,311 February and 123,624 June rows, not on 2026-06-11 (already on the A12 layer); the
      LOBT clip is re-applied and a negative result keeps the old value. Evidence: with 2025 February and June held
      out, A12 vs P12 −1.434 s [−1.997, −1.056] (`docs/appear_tier_2026-10-05/results.md` §21). Details:
      `docs/FINAL_SUBMISSION.md` §2.5.
3. **Combining.** `scripts/final/assemble_final_v48.py` combines the months and adds the increment.
   - It rebuilds these steps from the committed chain files (`handoff/2026-10-13/inputs/orig_chain_blend_adsb2_ovl.parquet`,
     `blend_A12_ovl.parquet`), v48 and the increment table.
   - `--febjun-appear handoff/2026-10-13/feb_jun_appear_increment_on_blend5.parquet` writes **the uploaded file
     `c0d4a143…` byte for byte** (0 rows differ; rerun 2026-10-14).
   - Without it, the default gives the base `be8d89dc…` (`25f6246`) and `--june-layer a12` gives `c7ffa31e…`
     (`558c736`), also byte for byte.
   - The members, blend, ADS-B layers, overlays and the increment table are taken as committed, not rebuilt. The
     increment table can be recomputed from committed files (`docs/FINAL_SUBMISSION.md` §5.3).
4. **Gate:** `python -m prc.validate_final <file> --template final_submitting.parquet --ranking final_ranking.parquet
   --expect-md5 <md5>`. It prints PASS; the output is summarised in `docs/FINAL_SUBMISSION.md` §1.

**What is not in the file** (`docs/FINAL_SUBMISSION.md` §3):
- no "+1×δ" re-anchor of January/July;
- none of the later candidates on the branches: the laughing-cray "v2" to "v7a" files (including v6, `7e53142e…`),
  the reweight and LTFM variants, and the beautiful-davinci LIRF-gate variants.

**Hand-off notes.** The `handoff/<date>/` folders are dated working notes between parallel sessions. Several name a
"recommended" file, such as `handoff/2026-10-13/README.md` (`d0ce32b6…`, later `7e53142e…`) and
`handoff/2026-10-14/README.md` / `handoff/2026-10-13/CANDIDATES.md` (`b7e8298d…`). **Those files were not submitted.**
The uploaded file itself is marked withdrawn in `handoff/2026-10-13/CANDIDATES.md` (06:20 UTC on 10-13); it was chosen
on 10-14 all the same (`docs/FINAL_SUBMISSION.md` §7).

**v48's EDDF coverage-matched delta.**
- **The problem.** In 2026, adsb.lol has fewer than half the 2025 ground fixes per EDDF departure: 76–125 in 2025
  against 35 in 2026 (`handoff/2026-10-08/COV26_EDDF.md`).
- **The fix.**
  - The 2025 EDDF fixes were thinned to the 2026 density, counted per 100 m cell (`docs/appear_tier_2026-10-05/plan_cov26.md`,
    `cov_thin.py`).
  - The 12-month appear residual was retrained on them (A12thin).
  - The delta `c_A12thin(b) − c_A12(b)` was added on the 2026 EDDF rows only:
    `handoff/2026-10-08/cov26_eddf_delta_vs_v47.parquet`, 31,774 rows, mean −3.8 s.
- **CV.** On thinned EDDF rows, with January and July 2025 held out: EDDF −1.13 s [−1.73, −0.46], all rows −0.19 s
  [−0.39, −0.02] (`docs/appear_tier_2026-10-05/results.md` §18).
- **Leaderboard.** v48 put it on C15 together with the 12-month appear delta (`handoff/2026-10-08/ON_C15.md`). The two
  together moved the leaderboard from 234.4831 (v47) to 234.4099 (`docs/DEVELOPMENT_HISTORY.md` §1). The EDDF part
  was not scored on its own.
- **What was not adopted.**
  - Re-weighting the training months by coverage similarity was tested separately and rejected; no delta was built
    from it (`docs/covw_2026-10-08/README.md`).
  - Thinning at test time at airports whose coverage became denser was rejected too (`results.md` §20).

**Reproducibility, briefly** (details and limits: `docs/FINAL_SUBMISSION.md` §5–§6).
- **What can be reproduced exactly.** The final file, byte for byte, with `scripts/final/assemble_final_v48.py
  --febjun-appear …` from committed files.
- **What cannot.** The January/July rows can only be copied from v48. Running the documented chain on January/July
  differs from v48 by RMS 53.2 s (`handoff/2026-10-13/README.md`; `docs/final_phase_2026-10-11/IMPACT.md` §4(c)).
- **Rerunning the chain from raw data needs:**
  - the competition files, including the 2025 `training_*.parquet` that the overlay stage reads;
  - the adsb.lol archives, 2–4 GB per day (194.9 GB for the 57 Jan/Jul 2026 days downloaded);
  - fresh fits of every member, because `src/prc` does not save LightGBM or CatBoost models.
- **Absolute paths.** Several scripts hard-code the absolute paths of the machines they ran on:
  `/home/user/pb-work`, `/home/user/prc-final` and scratch directories. `docs/FINAL_SUBMISSION.md` §5.3 maps them to
  the committed copies:
  - member predictions: `handoff/2026-10-13/members/`;
  - lane-B deltas, LIRF q and the union features: `handoff/2026-10-13/inputs/`;
  - per-day ADS-B features: `handoff/2026-10-13/adsb_store/` and `handoff/2026-10-11/adsb/`;
  - residual models: `docs/final_2026-10-11/adsb_models/`.

## 4. Environment

These are the steps from `docs/final_phase_2026-10-11/RUNBOOK_final_submit.md` §2. They were measured with uv 0.11.32
and Python 3.11.17.

```bash
uv venv -q .venv
uv pip install -q --python .venv/bin/python -e ".[dev]"
uv pip install -q --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -c "import prc, lightgbm, polars; print('env ok')"
.venv/bin/python -m pytest -q          # full suite; about 5-9 min
```

`make bootstrap` runs the same steps. It also downloads the competition data if the credentials are set.

## 5. Data access

The raw competition files are not in this repository. Some hand-off files carry per-row fields copied from them for
2026 rows, without labels; see `docs/final_phase_2026-10-11/PUBLIC_RELEASE_CHECKLIST.md`. The files come from the
organisers' OpenSky Network MinIO (S3) service. Credentials come from environment variables only (template: `.env.example`; `.env` is git-ignored):

| variable | meaning |
|---|---|
| `OSN_ACCESS_KEY` (or `OSN_ID`) | MinIO access key |
| `OSN_SECRET_KEY` (or `OSN_SECRET`) | MinIO secret key |
| `OSN_S3_ENDPOINT` | default `s3.opensky-network.org` |
| `OSN_COMPETITION_BUCKET` | `prc-2026-datasets` |
| `OSN_TEAM_BUCKET`, `PRC_TEAM_NAME` | only needed for submissions |

```bash
.venv/bin/python -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"   # all files
# final-phase files only, read-only (runbook §3):
.venv/bin/python -c "from pathlib import Path; from prc.io_minio import download_files; \
  [print(p) for p in download_files(bucket='prc-2026-datasets', dest=Path('data/raw/final'))]"
```

External data are fetched from public endpoints without credentials. Each source, its licence and the script that
fetches it are listed in `docs/external_data.md`.

## 6. Repository map

```
src/prc/         package: features, training (train.py, cv.py, mlp.py), pipeline, weather, OSM, OPDI (research),
                 submission gate and upload (validate_final.py, submit.py, io_minio.py), leaderboard reader
configs/         experiment recipes (yaml); the id of each is pinned in tests/test_*_params.py
scripts/         ADS-B ingest and learned residual (adsb_*.py), OSM fetch, OPDI ingest, blend builders
analysis/        post-model layers and audits (local_*.py): posjoin, LIRF overlays, LOBT clip, lane-B deltas
docs/            research notes, one folder per topic and date; SOLUTION_OVERVIEW.md, external_data.md,
                 DEVELOPMENT_HISTORY.md, features_log.md
docs/final_2026-10-11/      final-set chain: adsb_layer.py, apply_overlays.py, assemble_step.py, member configs, logs
docs/appear_tier_2026-10-05/ appear tier, final_adsb.py
handoff/<date>/  files passed between sessions: candidate files, deltas, member predictions, ADS-B stores
experiments/     append-only logs: experiments.jsonl (CV records), submissions.jsonl (every upload), pipeline.jsonl
tests/           pytest suite on synthetic data (no competition data needed)
sql/             early Snowflake load and profiling scripts; not used by the final chain
refs/            leaderboard API snapshots
```

## 7. External data and attribution

The full list is in `docs/external_data.md`. The model uses:
- **adsb.lol** `globe_history`: "Contains data from adsb.lol (globe_history), made available under the Open Database
  License (ODbL) 1.0."
- **OpenStreetMap**: "© OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0."
- **OurAirports**: public domain.
- **Iowa Environmental Mesonet** METAR: "Weather observations from the Iowa Environmental Mesonet (IEM) of Iowa State
  University."

Research only, not used in any submission or in the final file:
- **EUROCONTROL** OPDI v0.0.2 flight events and the PRU taxi-out additional time. Source: EUROCONTROL. Only aggregate
  results are kept in the tree. The per-row files derived from them were removed on 2026-10-14 under the EUROCONTROL
  terms (no modification without prior written permission); they remain in git history (`docs/external_data.md`).

The files under `handoff/**/adsb_store/`, `handoff/2026-10-11/adsb/`, `handoff/2026-10-13/june_adsb_for_adsb_layer/`
and the stand tables derived from OSM are derived databases of adsb.lol and OSM. They are made available under ODbL
1.0 (see `docs/external_data.md`).

**Data notice.** The repository contains no raw competition files. It does contain per-row files with fields
derived from the PRC Data Challenge 2026 dataset (EUROCONTROL PRC / OpenSky Network): mainly the per-departure ADS-B
feature and position files under `handoff/2026-10-13/` and `handoff/2026-10-11/adsb/learn/`, which carry times,
airport, runway and aircraft type of 2026 departures. No licence for that dataset had been published
at the time of release. If the organisers require these files to be removed, please contact the team
(details: `docs/final_phase_2026-10-11/PUBLIC_RELEASE_CHECKLIST.md` §3.2).

## 8. Development history

`docs/DEVELOPMENT_HISTORY.md` lists, from `experiments/submissions.jsonl`:
- which leaderboard submissions were model updates and which were diagnostics or A/B checks;
- which rules were derived by inspecting ranking-set rows.

## 9. Licence

- Code: GNU General Public License v3.0 (`LICENSE`).
- Vendored third-party code: `docs/research_2026-10-05_aobt/kpi/pylib/` holds openpyxl 3.1.5 and et_xmlfile 2.0.0.
  Both are MIT; their licence files are next to them. They are used by one research script only.
- Data: see §7 and `docs/external_data.md`.

## 10. Results

Only numbers that are in committed documents appear here; each row names its source. The ΔRMSE values are in
seconds, and negative means better. The CIs are the 95 % bootstrap intervals that each source states. These are
2025 offline measurements: each row compares one change against the base of its day, so the rows **do not add up**.

**Abbreviations.**
- *pooled OOF*: the 2025 out-of-fold predictions of the `jan`, `jul` and `forward` month folds, 672,428 rows.
- *Jan+Jul ADS-B CV*: Jan + Jul 2025, 344,419 rows, leave-days-out (5 folds of days), day-block bootstrap.
- `R = docs/competitor_research_2026-09-21/results/`, `M = docs/research_2026-10-05_measure/`,
  `A = docs/appear_tier_2026-10-05/results.md`.

| stage | change measured | ΔRMSE [95 % CI] | data | source |
|---|---|---|---|---|
| member p2o (plain target) | blend H2 → H2 + p2o, weight 0.5. Alone, p2o is worse than the anchored model by +0.93 [+0.59, +1.31] | −0.78 [−1.00, −0.60] | pooled OOF | `R/plainy-blend.md` |
| member c2 (CatBoost) | + c2, equal weights 1/3. Alone +0.94 [+0.49, +1.56] | −1.75 [−2.08, −1.47] | pooled OOF | `R/catboost-blend.md` |
| lever L2 (forward NM windows) | in all three members | −0.32 [−0.45, −0.20] | pooled OOF | `R/blend-l2.md` |
| member pao (per-airport) | blend + pao. Alone vs a2: +0.00 [−0.54, +0.52] | −0.77 [−0.92, −0.64] | pooled OOF | `R/per-airport-member.md` |
| retune of the anchored member | blend `hybrid_blend_l2_pa` → `…_opt`. The member alone gains −2.41 [−2.85, −2.02] | −0.89 [−1.10, −0.72] | pooled OOF | `R/optuna-retune.md` |
| member m4 (MLP) | + m4, equal weights 1/4. Alone it is worse than M1 by +5.59 [+4.67, +6.69] | −0.75 [−0.93, −0.58] | pooled OOF | `R/mlp-member.md` |
| lever L6 (gate retention) | one anchored member, `ret_base` → `ret_l6` | −0.76 [−0.95, −0.56] | `forward` fold, 328,009 rows | `M/l6_verdict.txt`, `M/l6_plan.md` |
| cmix (LIRF copy mixture in a2o) | a2o member. The strict verdict is REJECT: only 1 of 10 airports improves | −0.49 [−0.83, −0.15] | pooled OOF | `M/cmix_verdict.txt` |
| lane B: NM-blind (w 0.25) | non-LIRF unmatched rows (7,841): −7.73 [−20.41, −4.10] | whole frame −0.42 | pooled OOF | `docs/research_2026-10-06_nmblind/verdict.txt`, `plan.md` |
| lane B: ucmix (LIRF unmatched) | LIRF unmatched rows (537): −143.58 [−253.82, −37.74] | whole frame −1.60 [−2.95, −0.40] | pooled OOF | `docs/research_2026-10-07_ucmix/verdict.txt`, `plan.md` |
| learned ADS-B residual | on the H2 base: RMSE 316.97 → 302.94. Gain 8,697.9 s² [7,691.4, 9,828.8] | −14.03 | Jan+Jul ADS-B CV | `R/adsb-learned-residual.md` §2 |
| learned ADS-B residual, refit | on the R6-cbopt base: RMSE 311.705 → 298.596. Gain 8,000.6 s² [7,007.3, 9,060.8] | −13.11 | Jan+Jul ADS-B CV | `R/adsb-learned-residual.md` §14 |
| appear tier (4 months) | appear vs plain residual | −1.858 [−2.542, −1.346] | Jan+Jul ADS-B CV | `A` §1 |
| appear tier (12 months, A12) | vs the production mix of the day | −0.804 [−0.924, −0.692] | 2025, 12 months | `A` §12 |
| appear tier on held-out Feb+Jun 2025 | A12 vs P12 | −1.434 [−1.997, −1.056] | Feb, Jun 2025 held out | `A` §21 |
| EDDF coverage delta (in v48) | A12 trained on thinned EDDF fixes. EDDF rows −1.13 [−1.73, −0.46] | all rows −0.19 [−0.39, −0.02] | Jan+Jul 2025, held out | `A` §18 |
| LIRF matched copy overlay (0.4·q) | 2025 Jan+Jul check: gain 390 s² per row (MSE; Jan 15, Jul 375). It changes 26,277 Jan/Jul 2026 rows | — | Jan+Jul 2025 | `docs/posjoin_2026-10-05/deeper_checks.md` |
| LIRF unmatched copy band | 34 of 34 such 2025 rows are schedule copies | — | 2025 | `docs/SOLUTION_OVERVIEW.md` §4 |
| LOBT clip ±3606 s | the identity window holds on every matched 2025 row (2.06 M) | — | 2025 | `docs/SOLUTION_OVERVIEW.md` §4 |

The best pre-ADS-B blend on 2026-10-07 had a pooled OOF RMSE of 269.69 (`docs/research_2026-10-07_ucmix/plan.md`).

**Offline gains carry over to the leaderboard unevenly** (`docs/SOLUTION_OVERVIEW.md` §6):
- consistency fixes: about 2× their CV estimate;
- model-level additions: about 1×;
- ADS-B residual add-ons: 10–35 %.

For example, the 12-month appear tier delivered about 9 % of its CV gain on the board
(`docs/appear_tier_2026-10-05/results.md` §17). The fl_/seq_ train/test consistency fix gained −1.36 s on the board
(`docs/SOLUTION_OVERVIEW.md` §5).

**Leaderboard history** (January + July 2026, 344,841 departures). The full table, with the class of every
submission (model update, A/B, diagnostic, few-row rule), is in
[`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) §1.

| version | what changed | leaderboard RMSE |
|---|---|---|
| v1 | baseline | 345.3520 |
| v4 | two clocks, weather, linear trees | 305.0270 |
| v10 | model B reparameterised on the schedule at LIRF | 284.3584 |
| v17 | ADS-B pushback override (rule) | 272.1788 |
| v21 | learned ADS-B residual replaces the rule | 254.7657 |
| v33 | plain-target, CatBoost, L2, per-airport and retuned members | 250.9496 |
| v34 | positional ADS-B overlay on 430 unmatched rows (a few-row rule; see `DEVELOPMENT_HISTORY.md` §2.2) | 246.7493 |
| v39 | appear tier | 239.5208 |
| v42 | C12 batch | 236.4369 |
| v47 | C15: per-member consistency fix, 3-seed L6 deltas, ucmix | 234.4831 |
| **v48** | C15 + 12-month appear + EDDF coverage delta: **the January/July rows of the final file** | **234.4099** |
| v50 | check on 2026-10-14: the Jan/Jul part of laughing-cray v6 (`janjul_only_final_v6_all5.parquet`, `55a91f49…`) | 234.5741 |
| v51 | control on 2026-10-14: v48 resubmitted unchanged (`8ee6679f…`) | 234.4099 |

v51 (v48 unchanged) scored exactly v48's original 234.4099, so the board had not moved, and v6's Jan/Jul rows are
0.1642 worse than v48's. That is why the final file keeps v48's Jan/Jul rows (`docs/FINAL_SUBMISSION.md` §4). The v50/v51 upload
lines are in `experiments/submissions.jsonl`; the scores were read from the leaderboard.

Up to v49, the ledger has 50 uploads under 48 version numbers. 12 are classed as diagnostics (two of them also carry
another class) and 3 as A/B checks (`docs/DEVELOPMENT_HISTORY.md` §1). v50 and v51 came after that count.

**The final file** (`c0d4a143…`, uploaded 2026-10-14 10:30:29Z) has no score in the repository: the scorer's result
file says `"status": "Succeeded"` and `"used_pairs": 670790` and contains no score
(`docs/final_phase_2026-10-11/final_result_2026-10-14.json`). Its January/July rows are v48's.

## 11. Documentation map

| topic | document |
|---|---|
| **the final file: identity, recipe, reproduction, limits** | [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) |
| index of `docs/` | [`docs/README.md`](docs/README.md) |
| pipeline map (state at v47) | `docs/SOLUTION_OVERVIEW.md` |
| leaderboard use, every submission, row-level rules | `docs/DEVELOPMENT_HISTORY.md` |
| external data, licences, fetch scripts | `docs/external_data.md` |
| upload runbook; release checklist; final-phase impact | `docs/final_phase_2026-10-11/RUNBOOK_final_submit.md`, `PUBLIC_RELEASE_CHECKLIST.md`, `IMPACT.md` |
| final-set chain scripts (ADS-B layer, overlays, assembly of later variants) | `docs/final_2026-10-11/` (`adsb_layer_README.md`, `overlays_README.md`) |
| final-file assembly | `scripts/final/assemble_final_v48.py` |
| member p2o (plain target) | `docs/competitor_research_2026-09-21/results/plainy-blend.md` |
| member c2 (CatBoost), and its retune | `docs/competitor_research_2026-09-21/results/catboost-blend.md`, `catboost-retune.md` |
| lever L2 in the members | `docs/competitor_research_2026-09-21/results/blend-l2.md` |
| member pao (per-airport) | `docs/competitor_research_2026-09-21/results/per-airport-member.md` |
| retune of the anchored member | `docs/competitor_research_2026-09-21/results/optuna-retune.md` |
| member m4 (MLP) | `docs/competitor_research_2026-09-21/results/mlp-member.md` |
| L6 gate retention, cmix | `docs/research_2026-10-05_measure/` (`l6_plan.md`, `l6_verdict.txt`, `cmix_verdict.txt`, `copy_plan.md`) |
| lane B: NM-blind, ucmix | `docs/research_2026-10-06_nmblind/`, `docs/research_2026-10-07_ucmix/` |
| learned ADS-B residual | `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md` |
| appear tier, 12-month refit, EDDF coverage delta | `docs/appear_tier_2026-10-05/results.md`, `handoff/2026-10-08/COV26_EDDF.md`, `ON_C15.md` |
| coverage-weighted training (rejected) | `docs/covw_2026-10-08/README.md` |
| positional ADS-B join and the v35 overlays | `docs/posjoin_2026-10-05/README.md`, `deeper_checks.md` |
| LIRF label classes | `docs/lirf_wrap_2026-10-06/README.md` |
| train/test consistency audit; ADS-B coverage audit | `docs/shift_2026-10-06/`, `docs/adsb_audit_2026-10-07/README.md` |
| validation rule and protocol | `docs/competitor_research_2026-09-21/results/14-validation-design.md`, `docs/SOLUTION_OVERVIEW.md` §6 |
