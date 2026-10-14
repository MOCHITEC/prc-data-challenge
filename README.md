# PRC Data Challenge 2026: taxi-out time prediction (team resilient-hedgehog)

Code and documentation of team **resilient-hedgehog** (MOCHITEC Inc.) for the
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) of the EUROCONTROL Performance Review
Commission and the OpenSky Network. Licence: GNU GPLv3 (`LICENSE`).

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
   (`final_adsb.py` for the final set).
6. **Overlays**, applied after the residual layer by `docs/final_2026-10-11/apply_overlays.py`:
   - lane-B deltas: NM-blind and ucmix;
   - LIRF matched schedule-copy overlay: `p' = p + 0.4·q·(MVT − SCHED − p)`, where q comes from a 2025 classifier;
   - LIRF unmatched copy band: 7 to 13.5 h late, predict `MVT − SCHED`;
   - LIRF unmatched time-zone class: per-prefix q blend;
   - **LOBT clip** last: project onto `[MVT − LOBT − 3606, MVT − LOBT + 3606]`. This window holds on every matched
     2025 row.

How each piece was validated: `docs/SOLUTION_OVERVIEW.md` §6 and the per-topic folders under `docs/`.

## 3. The final file and how it was produced

**The file to upload** is named in `docs/final_phase_2026-10-11/RUNBOOK_final_submit.md`:

| | |
|---|---|
| file | `handoff/2026-10-13/final_from_v48.parquet` **as committed in 558c736** on `claude/laughing-cray-vvwh0k` |
| md5 | `c7ffa31edc2a4de5f1837682d90e14cb`, 2,145,277 bytes, 670,790 rows |
| how to get it | `git show 558c736:handoff/2026-10-13/final_from_v48.parquet > final_from_v48.parquet` |

The file was later replaced at the same path (25f6246, md5 `be8d89dc…`) and deleted at the branch tip (6adbcac). It
is reachable only through the commit above.

**Recipe** (the laughing-cray hand-off: `handoff/2026-10-13/README.md`, top sections, as of 558c736):

1. **Members.** Each member was trained on the final raw data with `python -m prc.pipeline --config <member config>
   --skip-cv --raw-dir data/raw_final --work-dir data/work_final`. The code was peaceful-babbage commit `6a97bf0`
   (contained in this branch). Members: a2o (3 seeds), pao (3), p2o (3), m4 (3) and c2 (1 seed); configs in the
   table above.
   - On matched rows, the `fl_*` / `seq_*` columns of the feature cache were replaced by values built on the
     2025 + 2026 union (`analysis/local_audit_fix2026.py`). This is the train/test consistency fix of
     `docs/SOLUTION_OVERVIEW.md` §5.
   - `prc.pipeline.load_raw` matches `ranking.parquet` and `submitting.parquet` by exact name. So `data/raw_final`
     held the final files under those names. The commands that built that directory are not recorded.
2. **Blend.** Matched rows: `((a2o + pao)/2 + p2o + c2 + m4) / 4`. Unmatched rows: model B. The result is committed as
   `handoff/2026-10-13/pre_adsb_base_blend5_final.parquet`.
3. **ADS-B residual layer.**
   - February rows: `docs/final_2026-10-11/adsb_layer.py apply`.
   - June rows: `docs/appear_tier_2026-10-05/final_adsb.py` with the store `handoff/2026-10-11/adsb`. This was the
     "A12" layer; June's own layer did not exist yet.
   - The own layer has no appear term. So in this file the February rows carry no appear increment. The later
     variants add it to February and June (a37e5bb).
4. **Overlays and LOBT clip** on February and June rows:
   `docs/final_2026-10-11/apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`. See
   `docs/final_2026-10-11/overlays_README.md` and `laneB_deltas.py`.
5. **January and July rows** were copied unchanged from v48: `handoff/2026-10-08/C15_appear12_cov26eddf.parquet`,
   md5 `8ee6679f…`, leaderboard 234.4099.
   - v48 itself is the end of a chain of per-submission delta layers, v33 → v48. See `docs/SOLUTION_OVERVIEW.md`
     §7, `docs/batch_2026-10-06/README.md`, `handoff/2026-10-08/ON_C15.md` and
     `docs/final_2026-10-11/adsb_layer_README.md` §1.
6. **Combining** the Jan/Jul rows of v48 with the Feb/Jun chain rows into one file was not committed as a script.
   The file first appears as `cand_v4a_v48JanJul_oursFeb_A12Jun.parquet` (0220c69) and was renamed in 558c736.
   `docs/final_2026-10-11/assemble_step.py` (8777c62, later) implements the same combination for the later
   variants.
7. **Gate:** `python -m prc.validate_final <file> --template final_submitting.parquet --ranking
   final_ranking.parquet --expect-md5 <md5>`. It passed on 10-13; the output is in the runbook §5.

**Later candidates** exist on the branches. They are not the runbook's target. Their recipes are in their READMEs.

| file | md5 | recipe |
|---|---|---|
| `handoff/2026-10-13/final_from_v48.parquet` @ 25f6246 (June on the own ADS-B layer) | `be8d89dc…` | `handoff/2026-10-13/README.md` (update 02:30) |
| `handoff/2026-10-13/final_recommended.parquet` = `final_v4_cb5_mlp5.parquet` | `d0ce32b6…` | same README, section "★★ 最終版"; `docs/final_2026-10-11/assemble_step.py`, `derived_members.py` |
| `handoff/2026-10-14/final_v4_gatefull_plus1delta.parquet` (and 3 variants) | `b7e8298d…` | `handoff/2026-10-14/README.md`; `docs/final_phase_gate/` |

Which file was finally uploaded is recorded in `experiments/submissions.jsonl` by `prc.submit --final`.

**Honest note on reproducibility.** The final chain was run by several working sessions, each on its own branch and
machine. Those branches are merged here (`docs/final_phase_2026-10-11/PUBLIC_RELEASE_CHECKLIST.md` lists them).
- Several scripts hard-code the absolute paths of those machines: `/home/user/pb-work`, `/home/user/prc-final` and
  scratch directories. The path table at the end of `handoff/2026-10-13/README.md` maps them to the committed copies.
- The committed hand-off folders hold the intermediate files:
  - member predictions: `handoff/2026-10-13/members/`;
  - lane-B deltas, LIRF q and the union features: `handoff/2026-10-13/inputs/`;
  - per-day ADS-B features: `handoff/2026-10-13/adsb_store/` and `handoff/2026-10-11/adsb/`;
  - residual models: `docs/final_2026-10-11/adsb_models/`.
- Rerunning from raw data needs three things:
  - the competition files;
  - the adsb.lol archives, 2–4 GB per day (194.9 GB for the 57 Jan/Jul 2026 days downloaded);
  - fresh fits of every member, because `src/prc` does not save LightGBM or CatBoost models.
- Recipe documents: `handoff/2026-10-13/README.md`, `handoff/2026-10-14/README.md`, `docs/final_2026-10-11/README.md`,
  `docs/final_phase_2026-10-13/README.md`, `docs/final_phase_2026-10-11/IMPACT.md` §4(c), `RUNBOOK_final_submit.md`.

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

The files under `handoff/**/adsb_store/`, `handoff/2026-10-11/adsb/`, `handoff/2026-10-13/june_adsb_for_adsb_layer/`
and the stand tables derived from OSM are derived databases of adsb.lol and OSM. They are made available under ODbL
1.0 (see `docs/external_data.md`).

## 8. Development history

`docs/DEVELOPMENT_HISTORY.md` lists, from `experiments/submissions.jsonl`:
- which leaderboard submissions were model updates and which were diagnostics or A/B checks;
- which rules were derived by inspecting ranking-set rows.

## 9. Licence

- Code: GNU General Public License v3.0 (`LICENSE`).
- Vendored third-party code: `docs/research_2026-10-05_aobt/kpi/pylib/` holds openpyxl 3.1.5 and et_xmlfile 2.0.0.
  Both are MIT; their licence files are next to them. They are used by one research script only.
- Data: see §7 and `docs/external_data.md`.
