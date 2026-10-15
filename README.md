# PRC Data Challenge 2026: taxi-out time prediction (team resilient-hedgehog)

Code, notes and final prediction file of team **resilient-hedgehog** (MOCHITEC Inc.) for the
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) of the EUROCONTROL Performance Review
Commission and the OpenSky Network. The task is to predict the taxi-out time of departures at 10 European airports.
The model splits departures by whether the Network Manager off-block time `AOBT_3` is known. Where it is, five
members (LightGBM, CatBoost, MLP) are blended; four of them predict the offset from the proxy `MVT − AOBT_3`. The
other rows get a separate model. A LightGBM trained on adsb.lol ADS-B tracks then corrects the rows that adsb.lol
covers, and rules derived from 2025 data handle label artefacts at LIRF and clip each prediction to a window around
`MVT − LOBT`. The January/July rows of the final file are the team's best leaderboard submission (v48); the
February/June rows come from a rebuild of that pipeline on the final data. The final file can be re-assembled byte
for byte by one script from committed intermediate files and the organisers' final-phase files (§4); earlier stages
can only partly be rebuilt (§5).

## At a glance

| | |
|---|---|
| **Task** | predict `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` (s) for departures at 10 airports; metric RMSE (§1) |
| **Final file** | `handoff/2026-10-13/final_v5_plus_febjun_appear.parquet`, added in commit `ea0e0a8`; md5 **`c0d4a1439862e68852b65f6854322a29`**; 670,790 rows; 2,151,179 bytes ([`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) §1) |
| **Upload** | as `resilient-hedgehog_final.parquet`; bucket `last_modified` **2026-10-14T10:30:29Z**, 30 min 29 s after the stated deadline of 10:00Z. The deadline wording, what the bucket held at 10:00Z and the organisers' statement on what will be considered: [§3, "The deadline and the late upload"](#the-deadline-and-the-late-upload). Acceptance is the organisers' decision |
| **Scorer** | result file written 2026-10-14T10:30:38Z: `"status": "Succeeded"`, `"used_pairs": 670790`. It contains no score ([copy](docs/final_phase_2026-10-11/final_result_2026-10-14.json)) |
| **Best public leaderboard score** | v48: RMSE **234.4099** on January + July 2026 (344,841 departures). The final file's January/July rows are v48's, unchanged. Board history: §8 and [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) §1 |
| **Leaderboard use** | besides model updates, the uploads include diagnostic probes, A/B checks and few-row rules; each upload is classed in [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md). Probe files prepared on 2026-10-14 were not uploaded (§9) |
| **Reproduction** | byte for byte from committed intermediate files and the organisers' final-phase files (§4); what a full rebuild needs and what cannot be regenerated (§5) |
| **Licence** | GNU GPLv3 ([`LICENSE`](LICENSE)); external data and their licences in §7 |

## Contents

1. [Task and metric](#1-task-and-metric)
2. [Method overview](#2-method-overview)
3. [The final file](#3-the-final-file)
4. [Reproduce the final file](#4-reproduce-the-final-file)
5. [Full rebuild and its limits](#5-full-rebuild-and-its-limits)
6. [What is ours and what came from elsewhere](#6-what-is-ours-and-what-came-from-elsewhere)
7. [Data and licences](#7-data-and-licences)
8. [Results](#8-results)
9. [Development history and leaderboard use](#9-development-history-and-leaderboard-use)
10. [Repository map](#10-repository-map)
11. [Documentation map](#11-documentation-map)
12. [Licence and acknowledgements](#12-licence-and-acknowledgements)

## 1. Task and metric

- **Target.** `TAXITIME_SEC_mvt = MVT_TIME − BLOCK_TIME` (seconds) for departures at 10 airports: EDDF, EDDM, EGLL,
  EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.
- **Training data.** The organisers' 2025 movements (`training_2025-MM-01_*.parquet`, all 12 months), with
  the NM flight record joined (`AOBT_3_flt`, `LOBT_flt`, …).
- **Leaderboard set.** January and July 2026 (`ranking.parquet`, 344,841 departures).
- **Final set.** January, February, June and July 2026 (`final_ranking.parquet` / `final_submitting.parquet`,
  670,790 departures). The final file is `resilient-hedgehog_final.parquet`.
- **Metric.** RMSE over all rows.

### Glossary

| term | meaning in this repository |
|---|---|
| MVT | `MVT_TIME_UTC_mvt`, the time of the movement; the organisers call it "(best available) movement time" (`docs/investigation_2026-09-16/hypotheses/external.md`) |
| BLOCK | `BLOCK_TIME_UTC_mvt`, the block time of the same movement record. The target is `y = MVT − BLOCK` |
| SCHED | `SCHED_TIME_UTC_mvt`, the scheduled time of the movement record |
| NM | the Network Manager. Its flight record is joined to each movement (the `*_flt` columns) |
| AOBT_3 | `AOBT_3_flt`, the off-block time in the NM flight record. The proxy is `MVT − AOBT_3` |
| LOBT | `LOBT_flt`, a further off-block time of the NM flight record. On every matched 2025 row `y` lies within `MVT − LOBT ± 3606 s`, which gives the LOBT clip (`docs/SOLUTION_OVERVIEW.md` §4) |
| matched / unmatched lane | rows with / without `AOBT_3` (about 98.5 % / 1.5 %). Matched rows get the member blend, unmatched rows model B (§2) |
| OOF | out-of-fold predictions on 2025 rows. *Pooled OOF* means the `jan`, `jul` and `forward` folds, 672,428 rows (§8) |
| a2o, pao, p2o, c2, m4 | the five matched-lane members: tuned anchored LightGBM, per-airport anchored LightGBM, plain-target LightGBM, CatBoost, MLP. "Anchored" means the member predicts `y − (MVT − AOBT_3)` (`docs/METHOD.md` §3) |
| A12, P12, A12thin | ADS-B residual models fitted on 12 months of 2025: with the appear-tier columns (A12), without them (P12), and A12 retrained on EDDF fixes thinned to the 2026 density (A12thin). `c_X` is the correction of model X (`docs/METHOD.md` §6) |
| C12, C15 | the team's names for the candidates submitted as v42 (C12) and v47 (C15). v48 = C15 + the 12-month appear and EDDF coverage terms (`docs/FINAL_SUBMISSION.md` §2.1) |
| δ | in "+1×δ": the per-row change of `MVT_TIME_UTC_mvt` between `ranking.parquet` and `final_ranking.parquet` on the Jan/Jul rows, up to ±12 s (`docs/DEVELOPMENT_HISTORY.md` §2.6) |
| q | in the LIRF rules: the estimated chance that a row's label is a schedule copy (`y ≈ MVT − SCHED`) or belongs to a given label class, from 2025 rates or a 2025 classifier (`docs/METHOD.md` §7, `docs/DEVELOPMENT_HISTORY.md` §2.3) |
| laughing-cray, beautiful-davinci, peaceful-babbage, model-b | short names of the parallel working branches `claude/<name>-…` (for example `claude/laughing-cray-vvwh0k`) and of the work done on them ("thread") |
| v1–v51 | leaderboard submission numbers (`docs/DEVELOPMENT_HISTORY.md` §1). The labels "v2" to "v7a" in hand-off notes name other files (`docs/FINAL_SUBMISSION.md` §3) |

## 2. Method overview

The diagram shows how the rows of the final file were produced. Every box is described in
[`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) §2 (rows per month, §2.2 steps 1–5, §2.3 for 2026-06-11,
§2.5 for the appear increment). The model itself, with configs, features and formulas, is in
[`docs/METHOD.md`](docs/METHOD.md).

```mermaid
flowchart TD
    IN["Final set, 670,790 departures, Jan Feb Jun Jul 2026"] --> MON{"Month"}
    MON -->|"Jan and Jul, 344,841 rows"| V48["Leaderboard submission v48, copied unchanged"]
    MON -->|"Feb and Jun, 325,949 rows"| LANE{"NM off-block time AOBT_3 present"}
    LANE -->|"yes, matched lane"| MEM["Five members. a2o, pao, c2, m4 predict y − (MVT − AOBT_3), p2o predicts y"]
    LANE -->|"no, unmatched lane"| MB["Model B, LightGBM. At LIRF the target is relative to the schedule"]
    MEM --> BL["Blend. Mean of p2o, c2, m4 and the average of a2o and pao"]
    BL -->|"other days"| ADSB["Learned ADS-B residual layer, rows seen by adsb.lol only"]
    MB -->|"other days"| ADSB
    BL -->|"2026-06-11 only"| A12["A12 residual layer for the 6,099 rows of 2026-06-11, appear tier included"]
    MB -->|"2026-06-11 only"| A12
    ADSB --> OVL["Overlays. Lane-B deltas NM-blind and ucmix, LIRF copy band, time-zone class, matched copy overlay"]
    A12 --> OVL
    OVL --> CLIP["LOBT clip to MVT − LOBT ± 3606 s"]
    CLIP --> BASE["Floor at 0, round. Base file be8d89dc"]
    BASE --> APP["Appear increment c_A12 − c_P12, c_A12thin − c_P12 at EDDF; 06-11 skipped; changes 226,935 Feb and Jun rows, then the LOBT clip again"]
    APP --> FIN["Final file c0d4a143, 670,790 rows"]
    V48 --> FIN
```

1. **Lanes.** About 98.5 % of rows have the NM off-block time `AOBT_3` (matched lane). Over all four months of the
   final set, 661,713 rows are matched and 9,077 are not (`docs/SOLUTION_OVERVIEW.md` §1, `docs/FINAL_SUBMISSION.md`
   §2.2); the diagram's lane split concerns the February/June rows only.
2. **Members and blend.** All members but p2o predict `y − (MVT − AOBT_3)` (`src/prc/train.py`, `matched_target`).
   Blend: `((a2o + pao)/2 + p2o + c2 + m4) / 4`. Unmatched rows: model B, a LightGBM whose target at LIRF is
   `y − (MVT − SCHED)`.
3. **Learned ADS-B residual layer.** A LightGBM fitted on 2025 to the blend's residual, using ground-movement
   features from adsb.lol traces (push, first move, stand dwell, stops, positions) joined by callsign or by position.
   Clipped to ±1800 s; 0 at LTFM. 2026-06-11 uses the A12 layer, because the ingest of this layer had no adsb.lol
   data for that day.
4. **Overlays and LOBT clip.** Lane-B deltas (NM-blind, ucmix), LIRF rules fitted on 2025 rows (copy band, time-zone
   class, copy overlay), then a projection onto `[MVT − LOBT − 3606, MVT − LOBT + 3606]`, a window that holds on every
   matched 2025 row.
5. **Appear increment.** The appear tier describes where and when the aircraft is seen relative to its own stand.
   February/June get `c_A12 − c_P12` (`c_A12thin − c_P12` at EDDF) at the end; the LOBT clip is re-applied and a
   negative result keeps the old value.
6. **January/July = v48**, built from the same kinds of stages on the leaderboard set as a chain of per-submission
   delta files v33 → v48 that no single script builds. It carries the 12-month appear and EDDF coverage terms.

Details per step: `docs/METHOD.md` §2–§8. Validation: `docs/SOLUTION_OVERVIEW.md` §6; numbers: §8.

## 3. The final file

**The authoritative description is [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md).** Its §1 defines the
file's identity: source commit, md5, rows and validation result. This section is a summary.

- **The file.** `handoff/2026-10-13/final_v5_plus_febjun_appear.parquet`, md5 **`c0d4a1439862e68852b65f6854322a29`**,
  2,151,179 bytes, 670,790 rows. It was added in **`ea0e0a8`** (branch `claude/beautiful-davinci-ehugk4`, contained
  in `main`), deleted in `9849a14` and restored byte for byte in `0d43558`; it is in the tree on `main`.
- **Its rows** (`docs/FINAL_SUBMISSION.md` §2):

  | rows | values | `FINAL_SUBMISSION.md` |
  |---|---|---|
  | Jan + Jul, 344,841 | v48 unchanged: `handoff/2026-10-08/C15_appear12_cov26eddf.parquet`, md5 `8ee6679f…`, public leaderboard 234.4099 | §2.1 |
  | Feb 144,158; Jun except 2026-06-11, 175,692 | the chain of §2 run on the final data, giving the base `be8d89dc…` (`handoff/2026-10-13/final_from_v48.parquet` as committed in `25f6246`), plus the appear increment on 103,311 Feb and 123,624 Jun rows | §2.2, §2.5 |
  | Jun 2026-06-11, 6,099 | the same chain with the A12 layer instead of our rebuilt layer; no increment | §2.3 |

  Earlier versions of this README (before `1d12abb`) named the base `be8d89dc…` as the final file. It was not
  uploaded.
- **Why this file** (`docs/FINAL_SUBMISSION.md` §4). On 2026-10-14 the Jan/Jul part of the laughing-cray v6 file
  scored 234.5741 on the leaderboard (v50), while v48 resubmitted unchanged scored 234.4099 (v51). So the Jan/Jul
  rows stay v48, and February/June get the appear increment so that they follow v48's recipe.
- **Not in the file**: the "+1×δ" re-anchor, the LIRF copy-overlay gate and all later candidates
  (`docs/FINAL_SUBMISSION.md` §3). Several hand-off notes name a "recommended" file; those files were not submitted
  (§10).
- **The upload.** `prc-2026-resilient-hedgehog/resilient-hedgehog_final.parquet`, by `prc.submit --final`; bucket
  `last_modified` **2026-10-14 10:30:29Z**, read-back md5 matched. The ledger lines (`"kind": "final"`) are at the end
  of `experiments/submissions.jsonl`. The scorer's result file (10:30:38Z) says `"status": "Succeeded"`,
  `"used_pairs": 670790`, and contains no score (`docs/final_phase_2026-10-11/final_result_2026-10-14.json`).

### The deadline and the late upload

- **The stated deadline.** The organisers'
  [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html) says "Final single submission is
  uploaded by 2026-10-14T12:00 CET (2026-10-14T12:00+02:00)". The deployed index page gave "2026-10-14T10:00Z" when
  read on 2026-10-11 (`docs/final_phase_2026-10-11/IMPACT.md` §1.2). The upload was at 10:30:29Z, **30 min 29 s
  after 10:00Z**.
- **The bucket at 10:00Z.** The only `resilient-hedgehog_final.parquet` in the bucket was an object of 2026-10-09
  16:02Z (uploader unknown), which the organisers' scorer had marked Failed. The upload replaced it
  (`docs/FINAL_SUBMISSION.md` §1).
- **What the organisers will consider.** On the challenge Discord on 2026-10-14 the organisers stated that only
  commits made before the deadline, and the latest `_final.parquet` uploaded before the deadline, will be considered.
  No copy of the message is in this repository.
- **The repository at the deadline.** The last commit before 10:00Z is `aedbc1f` (2026-10-14T08:55:19Z). The final
  file and its increment table are in its history (`ea0e0a8`, `4507ddd`) but not in its tree: both were deleted in
  `9849a14` (2026-10-13) and restored in `0d43558` (10:31:38Z). The assembler option that adds the increment
  (`--febjun-appear`, §4.3) was added in `5f90818` (10:38:06Z).
- **Acceptance** of the upload, and of anything committed after 10:00Z, is the organisers' decision.

## 4. Reproduce the final file

These steps rebuild the uploaded file byte for byte and validate it (runbook
`docs/final_phase_2026-10-11/RUNBOOK_final_submit.md` §2–§5; `docs/FINAL_SUBMISSION.md` §5.2). Run every command from
the repository root. What the assembler does not rebuild is in §5.

### 4.1 Environment

Measured with uv 0.11.32 and Python 3.11.17, 1 min 34 s (runbook §2); `pyproject.toml` requires Python ≥ 3.11, < 3.13.

```bash
git clone https://github.com/MOCHITEC/prc-data-challenge.git   # or --depth 1 (the packed history is 250 MB)
cd prc-data-challenge
uv venv -q .venv
uv pip install -q --python .venv/bin/python -e ".[dev]"
uv pip install -q --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -c "import prc, lightgbm, polars; print('env ok')"
.venv/bin/python -m pytest -q          # optional: full suite on synthetic data, about 5-9 min
```

The tests need no competition data (364 passed, 2 skipped after the file removal of 2026-10-14;
`docs/final_phase_2026-10-11/PUBLIC_RELEASE_CHECKLIST.md` §3.1, which also gives the 250 MB in §4). `make bootstrap`
runs the same steps and also downloads the competition data if the credentials are set; that download step needs
`OSN_COMPETITION_BUCKET=prc-2026-datasets` exported, otherwise it prints only "download failed".

### 4.2 Competition files

The raw competition files are not in this repository (§7). They come from the organisers' OpenSky Network MinIO (S3)
service. Credentials come from environment variables only (template: `.env.example`; `.env` is git-ignored):

| variable | meaning |
|---|---|
| `OSN_ACCESS_KEY` (or `OSN_ID`) | MinIO access key |
| `OSN_SECRET_KEY` (or `OSN_SECRET`) | MinIO secret key |
| `OSN_S3_ENDPOINT` | default `s3.opensky-network.org` |
| `OSN_COMPETITION_BUCKET` | `prc-2026-datasets` |
| `OSN_TEAM_BUCKET`, `PRC_TEAM_NAME` | only needed for submissions |

Export the credentials and `OSN_COMPETITION_BUCKET=prc-2026-datasets` first; never print the secrets.

```bash
export OSN_COMPETITION_BUCKET=prc-2026-datasets
# final-phase files only, read-only (runbook §3):
.venv/bin/python -c "from pathlib import Path; from prc.io_minio import download_files; \
  [print(p) for p in download_files(bucket='prc-2026-datasets', dest=Path('data/raw/final'))]"
```

- This writes `final_submitting.parquet`, `final_ranking.parquet` and `ranking.parquet` (132 MB) to `data/raw/final/`.
  That is the assembler's default directory. The validator has no default: it takes the paths explicitly
  (`--template` is required in `src/prc/validate_final.py`), as in §4.3. Runbook §3 lists the files' md5 values of
  2026-10-13.
- `download_all()` (`.venv/bin/python -c "from prc.io_minio import download_all; print(len(download_all()), 'files')"`,
  also `make download`) fetches every object of the bucket, including the 2025 training files, into `data/raw/`, not
  `data/raw/final/`. It raises `RuntimeError: OSN_COMPETITION_BUCKET is not set` if the variable is not exported.
  Only the full rebuild (§5) needs it.

### 4.3 Assemble and validate

```bash
.venv/bin/python scripts/final/assemble_final_v48.py --out outputs/final/final_v5_plus_febjun_appear.parquet \
  --febjun-appear handoff/2026-10-13/feb_jun_appear_increment_on_blend5.parquet \
  --reference handoff/2026-10-13/final_v5_plus_febjun_appear.parquet \
  --expect-md5 c0d4a1439862e68852b65f6854322a29
.venv/bin/python -m prc.validate_final outputs/final/final_v5_plus_febjun_appear.parquet \
  --template data/raw/final/final_submitting.parquet --ranking data/raw/final/final_ranking.parquet \
  --expect-md5 c0d4a1439862e68852b65f6854322a29
```

- **Assembler.** It reads four committed files and stops if any md5 differs: v48 (Jan/Jul), the own-layer chain
  output `handoff/2026-10-13/inputs/orig_chain_blend_adsb2_ovl.parquet` (Feb, Jun except 06-11), the A12 chain output
  `handoff/2026-10-13/inputs/blend_A12_ovl.parquet` (06-11) and the increment table. It picks the source per row,
  floors at 0, adds the increment, re-applies the LOBT clip from `final_ranking.parquet` and reverts negative results.
  Rerun on 2026-10-14, it wrote the same bytes as the committed and uploaded file. Without `--febjun-appear` it
  writes the base `be8d89dc…` (`25f6246`), and `--june-layer a12` gives `c7ffa31e…` (`558c736`), also byte for byte
  (`docs/FINAL_SUBMISSION.md` §5.2).
- **Validator.** It prints **PASS**. It fails on a schema, row-count or id mismatch with the template, on nulls,
  negatives or a constant vector, on a missing or extra month, and on a different md5 (docstring of
  `src/prc/validate_final.py`). For the final file it reports 66 rows above 10,800 s as a warning
  (`docs/FINAL_SUBMISSION.md` §1). Run on `handoff/2026-10-13/final_v5_plus_febjun_appear.parquet`, it checks the
  committed file itself.

## 5. Full rebuild and its limits

The assembler of §4.3 takes v48, the two chain files and the increment table as committed. Rebuilding further upstream
is possible for some stages and not for others. The commands and the map from the scripts' hard-coded paths to the
committed copies are in `docs/FINAL_SUBMISSION.md` §5.3; the limits in §6 there.

| stage | what it needs beyond the repository | committed result |
|---|---|---|
| members (Feb/Jun) | the organisers' 2025 training files and a fresh fit: `src/prc` saves no LightGBM or CatBoost model, and how close a refit comes was not measured | `handoff/2026-10-13/members/` |
| blend | — | `handoff/2026-10-13/pre_adsb_base_blend5_final.parquet` |
| ADS-B residual layer | the final ranking file; the script's paths mapped to the committed copies | models `docs/final_2026-10-11/adsb_models/`, features `handoff/2026-10-13/adsb_store/` |
| overlays and LOBT clip | the organisers' 2025 training files and the `prc.pipeline` feature cache | the two chain files in `handoff/2026-10-13/inputs/` |
| appear increment table | only `final_ranking.parquet`; recomputed on 2026-10-14, equal values on all 227,019 rows | `handoff/2026-10-13/feb_jun_appear_increment_on_blend5.parquet` |
| ADS-B features from raw traces | the adsb.lol archives: 194.9 GB for the 57 Jan/Jul 2026 days downloaded, 166.11 GB for Jan/Jul 2025; the total for 12 months of 2025 was not measured | the per-day features above, `handoff/2026-10-11/adsb/` |

- **January/July cannot be regenerated exactly.** They are copied from v48, the end of a chain of delta files. The
  documented chain differs from v48 by RMS 53.2 s on those rows, 43.4 s with the appear increment.
- **The months follow the same recipe only approximately** (different ADS-B layer chains; the increment added to the
  finished file), and no leaderboard measurement covers February/June.

## 6. What is ours and what came from elsewhere

The organisers' [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html) (read 2026-10-15)
sets this criterion for prizes: "The solution is original: Teams must use their own original solutions. Re-using any
existing implementation is only allowed if the original authors grant you the rights to use their solution and if
you made significant modifications to the algorithm or model. In particular, simply re-using existing code and
rewriting the data input and output mechanism is not sufficient. Adding parameters to the model and modifying filters
to match the specific peculiarities of the data, however, can be considered sufficient."

From the team's notes (`R` = `docs/competitor_research_2026-09-21/results/`):

- **Code.** Competitor repositories were read for ideas; the notes that read them state that no code was copied
  (`R/public-code-deep-dive-2026-09-30.md`, `docs/research_2026-10-05_repos/known_updates.md`), and the brief for
  measuring competitor ideas says "Never copy competitor code (GPL); ideas only"
  (`docs/research_2026-10-05_measure/PROTOCOL.md`). For the ADS-B extraction, ingest, features, residual, positional
  join, appear tier and final-set layer: "All of this code was written for this repository" (`docs/external_data.md`).
  The vendored third-party code recorded in the release checklist is openpyxl and et_xmlfile (MIT, one research
  script; §12). The models use LightGBM, CatBoost and scikit-learn.
- **Process.** Most notes are pre-registrations committed before the run, followed by results (`docs/README.md`);
  most changes were accepted by a pre-registered rule on the team's 2025 data (`docs/SOLUTION_OVERVIEW.md` §6). cmix
  was kept although its strict verdict was REJECT (§8).

**Components of the final file for which the notes name no outside source:**

| component | basis, per the notes |
|---|---|
| member m4 (MLP) | a function class different from the tree members (`R/mlp-member.md` §0.1) |
| ucmix | the team's error decomposition of its OOF predictions (`docs/research_2026-10-07_ucmix/plan.md`, "出所") |
| EDDF coverage-matched thinning | the team's audit of the 2025 → 2026 receiver-coverage change (`docs/appear_tier_2026-10-05/plan_cov26.md` §0, `docs/adsb_audit_2026-10-07/README.md`) |
| fl_/seq_ consistency fix | the team's audit of the 2025 → 2026 shift (`docs/shift_2026-10-06/README.md`) |
| LIRF copy band, time-zone class | derived from 2025 LIRF rows (`docs/lirf_wrap_2026-10-06/README.md`). That note also cites two public repositories that report the related "one-day" class, which is not in the final file |
| ADS-B residual: features, model, code | "The features, the model and the code are our own" (`R/adsb-learned-residual.md`); the idea of a learned residual came from outside (next table) |

**Components whose idea came from elsewhere:**

| component | source of the idea, per the notes |
|---|---|
| two lanes; anchored target `y − (MVT − AOBT_3)` | public README of `saurabhdxt-labs/prc-challenge` (team merry-quicksand per `docs/research_2026-10-05_repos/inventory.md`), in the prior-art note of 2026-09-10 (`git show 48bd4a2:docs/03_prior_art.md` §8, Japanese); `docs/competitor_research_2026-09-21/SUMMARY.md` §H |
| members p2o (plain target), c2 (CatBoost), pao (per-airport); lever L2; the retune | kind-mango's public repository, via `R/public-code-deep-dive-2026-09-30.md`; named as the source in `R/plainy-blend.md`, `R/catboost-blend.md`, `R/per-airport-member.md`, `R/feature-levers-L1-L4.md`, `R/optuna-retune.md` |
| lever L6 (gate retention) | other teams' congestion ideas, measured on the team's data (`docs/research_2026-10-05_measure/congestion.md`, Japanese) |
| cmix (LIRF copy mixture) | outgoing-boat's two-stage copy mixture (`docs/research_2026-10-05_measure/copy_plan.md`) |
| NM-blind blend | "Source: merry-mushroom v5" (`docs/research_2026-10-06_nmblind/plan.md`) |
| learned ADS-B residual | "The only outside fact used is …: one competitor shipped a learned ADS-B residual model. No competitor code was opened" (`R/adsb-learned-residual.md`) |
| appear tier | smart-jigsaw's public implementation, which takes the time an aircraft is first seen at its stand as off-block (`docs/research_2026-10-01/gap/HANDOFF_PROMPT.md` item 2, Japanese, "code for reference only, do not copy"); ours uses OSM stands and a 2026 ADS-B stand map as residual-model columns (`docs/appear_tier_2026-10-05/plan.md` §0) |
| LOBT clip ±3606 s | javidmardanov's public LOBT window rule, measured on the team's data before use (`docs/competitor_research_2026-09-21/plans/P5-physics-lower-bound.md` §0); outgoing-boat uses the same window (`docs/research_2026-10-05_repos/inventory.md`) |

These ideas were implemented in this repository; the notes state that no competitor code was copied (above). Each of
them except the two-lane split and the anchored target was measured on 2025 data before use; the measurements are in
`docs/METHOD.md` §9 (a selection in §8).

The tables do not classify the feature groups, model B's schedule target at LIRF or the positional ADS-B join; their
notes are listed in §11.

## 7. Data and licences

External data are fetched from public endpoints without credentials. The full list, with URLs, licences, fetch
scripts and joins, is [`docs/external_data.md`](docs/external_data.md).

| source | licence | status |
|---|---|---|
| adsb.lol `globe_history` ADS-B traces | ODbL 1.0 | final chain |
| OpenStreetMap (`aeroway=parking_position`, `aeroway=runway`) | ODbL 1.0 | final chain |
| OurAirports `airports.csv`, `runways.csv` | public domain | final chain |
| Iowa Environmental Mesonet ASOS/METAR | public domain, attribution requested | final chain |
| IANA time-zone database (through polars) | public domain | final chain |
| research only: Open-Meteo (CC BY 4.0); EUROCONTROL OPDI, PRU taxi-out additional time and ansperformance files | see `docs/external_data.md` (EUROCONTROL: attribution, non-commercial, no modification without permission) | not in any submission or the final file |

The model uses:
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
and the stand tables derived from OSM contain derived databases of adsb.lol and OSM. Their adsb.lol- and OSM-derived
content is made available under ODbL 1.0 (see `docs/external_data.md`). Many of the same files also carry columns
derived from the competition data (below); the ODbL statement does not cover those columns.

**Data notice.** The repository contains no raw competition file (`training_*`, `ranking`, `submitting`, `final_*`).
It does contain per-row fields derived from the PRC Data Challenge 2026 dataset (EUROCONTROL PRC / OpenSky Network),
in these files (`docs/final_phase_2026-10-11/PUBLIC_RELEASE_CHECKLIST.md` §3.2, audit of 2026-10-14):
- 412 parquet files (142.8 MB, 1,410,814 rows, all 2026 rows): the per-departure ADS-B feature and position files
  under `handoff/2026-10-13/adsb_store/`, `handoff/2026-10-13/june_adsb_for_adsb_layer/` and
  `handoff/2026-10-11/adsb/learn/`. Every one carries airport, movement time and `AOBT_3`; some also the scheduled
  time, runway and aircraft type;
- one CSV, `docs/research_2026-10-01/rollover/fingerprints/rank_candidates_ms8h.csv` (22 rows);
- `handoff/2026-10-13/adsb_store/rescat/codes.parquet` (airline and apron codes derived from the callsign, flight
  number, airport and stand).

No licence for the 2026 dataset had been published at the time of release. If the organisers require these files to
be removed, please open an issue on this repository.

## 8. Results

**Offline (2025).** A selection of the per-stage measurements; the full table, with the change measured in each row,
is `docs/METHOD.md` §9. ΔRMSE in seconds, negative is better, with the 95 % bootstrap CI that each source states. Each
row compares one change against the base of its day, so the rows **do not add up**. *Pooled OOF*: the `jan`, `jul`
and `forward` folds of 2025, 672,428 rows. *Jan+Jul ADS-B CV*: Jan + Jul 2025, 344,419 rows, leave-days-out.
`R = docs/competitor_research_2026-09-21/results/`, `M = docs/research_2026-10-05_measure/`,
`A = docs/appear_tier_2026-10-05/results.md`.

| stage | ΔRMSE [95 % CI] | data | source |
|---|---|---|---|
| member p2o (plain target) | −0.78 [−1.00, −0.60] | pooled OOF | `R/plainy-blend.md` |
| member c2 (CatBoost) | −1.75 [−2.08, −1.47] | pooled OOF | `R/catboost-blend.md` |
| member pao (per-airport) | −0.77 [−0.92, −0.64] | pooled OOF | `R/per-airport-member.md` |
| member m4 (MLP) | −0.75 [−0.93, −0.58] | pooled OOF | `R/mlp-member.md` |
| cmix (LIRF copy mixture in a2o) | −0.49 [−0.83, −0.15]; strict verdict REJECT (1 of 10 airports improves, `docs/research_2026-10-05_measure/cmix_verdict.txt`); kept in a2o | pooled OOF | `M/cmix_verdict.txt` |
| lane B: ucmix (LIRF unmatched) | whole frame −1.60 [−2.95, −0.40] | pooled OOF | `docs/research_2026-10-07_ucmix/verdict.txt` |
| learned ADS-B residual | −14.03 | Jan+Jul ADS-B CV | `R/adsb-learned-residual.md` §2 |
| appear tier (12 months, A12) | −0.804 [−0.924, −0.692] | 2025, 12 months | `A` §12 |
| appear tier on held-out Feb+Jun 2025 | −1.434 [−1.997, −1.056] | Feb, Jun 2025 held out | `A` §21 |
| EDDF coverage delta (in v48) | all rows −0.19 [−0.39, −0.02] | Jan+Jul 2025, held out | `A` §18 |
| LOBT clip ±3606 s | the window holds on every matched row (2.06 M) | 2025 | `docs/SOLUTION_OVERVIEW.md` §4 |

Offline gains carried over to the board unevenly: consistency fixes about 2× their CV estimate, model-level
additions about 1×, ADS-B residual add-ons 10–35 % (`docs/SOLUTION_OVERVIEW.md` §6).

**Leaderboard history** (January + July 2026, 344,841 departures). Scores as listed in the committed board snapshot
`refs/leaderboard_snapshots/2026-10-15T00Z_leaderboard.json`. Every submission, with its class, is in
[`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) §1.

| version | what changed | leaderboard RMSE |
|---|---|---|
| v1 | baseline | 345.3520 |
| v4 | two clocks, weather, linear trees | 305.0270 |
| v10 | model B reparameterised on the schedule at LIRF | 284.3584 |
| v17 | ADS-B pushback override (rule) | 272.1788 |
| v21 | learned ADS-B residual replaces the rule | 254.7657 |
| v33 | plain-target, CatBoost, L2, per-airport and retuned members | 250.9496 |
| v34 | positional ADS-B overlay on 430 unmatched rows (a few-row rule; see `docs/DEVELOPMENT_HISTORY.md` §2.2) | 246.7493 |
| v39 | appear tier | 239.5208 |
| v42 | C12 batch | 236.4369 |
| v47 | C15: per-member consistency fix, 3-seed L6 deltas, ucmix | 234.4831 |
| **v48** | C15 + 12-month appear + EDDF coverage delta: **the January/July rows of the final file** | **234.4099** |
| v50 | check on 2026-10-14: the Jan/Jul part of laughing-cray v6 (`handoff/2026-10-13/janjul_only_final_v6_all5.parquet`, `55a91f49…`) | 234.5741 |
| v51 | control on 2026-10-14: v48 resubmitted unchanged (`8ee6679f…`) | 234.4099 |

v51 scored exactly v48's 234.4099, so the board had not moved; v6's Jan/Jul rows are 0.1642 worse
(`docs/FINAL_SUBMISSION.md` §4). The final file itself has no score in the repository (§3).

## 9. Development history and leaderboard use

[`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) lists, from `experiments/submissions.jsonl`, which
leaderboard submissions were model updates and which were diagnostics or A/B checks, and which rules were derived by
inspecting ranking-set rows. In short:
- **Counts** (its §1). The ledger has 52 leaderboard uploads under 50 version numbers (v1–v51 without v24): 33 model
  updates, 14 diagnostics (in the ledger's words: probe, isolation, decomposition, test, check or control), 3 A/B
  checks and 4 few-row rules; two uploads carry two classes. Then came the final upload (§3). The scores are in the
  committed board snapshot `refs/leaderboard_snapshots/2026-10-15T00Z_leaderboard.json`.
- **Ranking-set inspection** (its §2).
  - Rule C7r of v41 (the LIRF "one-day" class) set q values for named 2026 flights, and the v41 board score was used
    to infer the labels of three specific ranking rows. C7r was then dropped; the final chain does not apply it. The
    two LIRF classes the chain keeps take their rates from 2025 (§2.3).
  - The positional ADS-B rule of v34 was motivated by EHAM snow days in the ranking set; its weights were set on 2025
    rows, and it is still in the chain as the `RULE` component of the ADS-B layer (§2.2; `docs/METHOD.md` §5).
  - Board scores were also read back into decisions: the lane and component isolation uploads (v9, v10, v13, v31,
    v43–v45), the v30 graft, the board-transfer ratios used in planning, v49, and v50 against v51 (§2.4).
  - The "+1×δ" re-anchor and the LIRF copy-overlay gate came in the final phase, after the final set was published
    on 2026-10-08; neither is in the final file (§2.6; `docs/FINAL_SUBMISSION.md` §3).
- **Probe and test files prepared on 2026-10-14 and not uploaded** (its §2.7).
  - `handoff/2026-10-14/probes/` holds probes P1–P6: copies of v48 with all rows of one cell (LTFM in January or
    July, or January de-icing rows) shifted by −40 or −100 s, plus a control. Its README is titled "Four tests of the
    mean shift on the old board (January, July) (user-approved, 2026-10-14 05:55 UTC)". By its design, the score of
    each probe gives the mean residual of v48 against the labels on that cell (`rbar = mean(y - v48)`), and
    `docs/final_phase_gate/probe_fix.py` and `docs/final_phase_gate/assemble_final.py` would then shift the same departures of the final
    file by that amount.
  - `handoff/2026-10-14/oldboard/` holds v48 plus the LIRF copy-overlay gate, and v48 plus the gate and a q^1.5
    re-weighting, to be compared with v48 on the board before choosing the final file.
  - None of these files was uploaded. The ledger has no line for them; the board and the team bucket hold no numbered
    files other than the team's v1–v23 and v25–v51 (`refs/leaderboard_snapshots/2026-10-15T00Z_leaderboard.json`,
    `refs/team_bucket/2026-10-15T01Z_objects.json`). The only regular uploads of 2026-10-14 were v50 (v6's Jan/Jul
    part) and v51 (v48 unchanged). The notes do not record why the probes were not sent.
- **Team rule A9** (2026-09-21) says that an upload "chosen to make the returned RMSE informative about the labels" is
  a label measurement and "should not be sent". The A/B and diagnostic uploads made after it, and what the notes say
  about them, are in its §3.
- **Dropped probes.** On 2026-10-11 two planned leaderboard probes were dropped because of the organisers' sentence
  "We will monitor submissions for attempts to learn from or exploit the ranking process" (its introduction).
- **Process facts** (its §3): two version collisions, an unrecorded 2026-10-09 final upload, and the choices that
  the notes label "post hoc".

## 10. Repository map

```
src/prc/         package: features, training (train.py, cv.py, mlp.py), pipeline, weather, OSM, OPDI (research),
                 submission gate and upload (validate_final.py, submit.py, io_minio.py), leaderboard reader
configs/         experiment recipes (yaml); the id of each is pinned in tests/test_*_params.py
scripts/         ADS-B ingest and learned residual (adsb_*.py), OSM fetch, OPDI ingest, blend builders;
                 scripts/final/assemble_final_v48.py rebuilds the final file (§4.3)
analysis/        post-model layers and audits (local_*.py): posjoin, LIRF overlays, LOBT clip, lane-B deltas
docs/            research notes, one folder per topic and date; FINAL_SUBMISSION.md, METHOD.md,
                 SOLUTION_OVERVIEW.md, external_data.md, DEVELOPMENT_HISTORY.md, features_log.md
docs/final_2026-10-11/      final-set chain: adsb_layer.py, apply_overlays.py, assemble_step.py, member configs, logs
docs/appear_tier_2026-10-05/ appear tier, final_adsb.py
handoff/<date>/  dated working notes and files passed between parallel sessions: candidate files, deltas,
                 member predictions, ADS-B stores (see below)
experiments/     append-only logs: experiments.jsonl (CV records), submissions.jsonl (every upload), pipeline.jsonl
tests/           pytest suite on synthetic data (no competition data needed)
sql/             early Snowflake load and profiling scripts; not used by the final chain
refs/            leaderboard API snapshots (the last of 2026-10-15) and the team bucket's object list
outputs/         two shell scripts of the final-phase member runs (outputs/ret_logs/); the assembler writes here
.claude/         CLAUDE.md, working rules for the coding-assistant sessions (in Japanese)
```

**About `handoff/`.** The `handoff/<date>/` folders are dated working notes that parallel sessions passed to each
other, mostly in Japanese, kept as written as the record (`docs/FINAL_SUBMISSION.md` §7).
- Several name a "recommended" file (`d0ce32b6…`, `7e53142e…`, `b7e8298d…`). **Those files were not submitted.** The
  uploaded file itself is marked withdrawn in `handoff/2026-10-13/CANDIDATES.md` (06:20 UTC on 10-13); it was chosen
  on 10-14 all the same.
- Where a hand-off note and `docs/FINAL_SUBMISSION.md` disagree about the final file, `docs/FINAL_SUBMISSION.md` is
  right; the upload ledger `experiments/submissions.jsonl` and the bucket override both. The parts of
  `handoff/2026-10-13/README.md` that concern the final file are translated there (§2.2, §5.3, §8).

## 11. Documentation map

| topic | document |
|---|---|
| **the final file: identity, recipe, reproduction, limits** | [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) |
| the model: lanes, members, ADS-B layer, appear tier, EDDF delta, overlays | [`docs/METHOD.md`](docs/METHOD.md) |
| index of `docs/` | [`docs/README.md`](docs/README.md) |
| pipeline map (state at v47) | [`docs/SOLUTION_OVERVIEW.md`](docs/SOLUTION_OVERVIEW.md) |
| leaderboard use, every submission, row-level rules | [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) |
| external data, licences, fetch scripts | [`docs/external_data.md`](docs/external_data.md) |
| upload runbook; release checklist; final-phase impact | [`RUNBOOK_final_submit.md`](docs/final_phase_2026-10-11/RUNBOOK_final_submit.md), [`PUBLIC_RELEASE_CHECKLIST.md`](docs/final_phase_2026-10-11/PUBLIC_RELEASE_CHECKLIST.md), [`IMPACT.md`](docs/final_phase_2026-10-11/IMPACT.md) |
| final-set chain scripts (ADS-B layer, overlays, assembly of later variants) | [`docs/final_2026-10-11/`](docs/final_2026-10-11/) ([`adsb_layer_README.md`](docs/final_2026-10-11/adsb_layer_README.md), [`overlays_README.md`](docs/final_2026-10-11/overlays_README.md)) |
| final-file assembly | [`scripts/final/assemble_final_v48.py`](scripts/final/assemble_final_v48.py) |
| members and levers: p2o, c2 (and its retune), L2, pao, retune of the anchored member, m4 | [`plainy-blend.md`](docs/competitor_research_2026-09-21/results/plainy-blend.md), [`catboost-blend.md`](docs/competitor_research_2026-09-21/results/catboost-blend.md), [`catboost-retune.md`](docs/competitor_research_2026-09-21/results/catboost-retune.md), [`blend-l2.md`](docs/competitor_research_2026-09-21/results/blend-l2.md), [`per-airport-member.md`](docs/competitor_research_2026-09-21/results/per-airport-member.md), [`optuna-retune.md`](docs/competitor_research_2026-09-21/results/optuna-retune.md), [`mlp-member.md`](docs/competitor_research_2026-09-21/results/mlp-member.md) |
| L6 gate retention, cmix | [`docs/research_2026-10-05_measure/`](docs/research_2026-10-05_measure/) (`l6_plan.md`, `l6_verdict.txt`, `cmix_verdict.txt`, `copy_plan.md`) |
| lane B: NM-blind, ucmix | [`docs/research_2026-10-06_nmblind/`](docs/research_2026-10-06_nmblind/), [`docs/research_2026-10-07_ucmix/`](docs/research_2026-10-07_ucmix/) |
| learned ADS-B residual | [`adsb-learned-residual.md`](docs/competitor_research_2026-09-21/results/adsb-learned-residual.md) |
| appear tier, 12-month refit, EDDF coverage delta | [`results.md`](docs/appear_tier_2026-10-05/results.md), [`COV26_EDDF.md`](handoff/2026-10-08/COV26_EDDF.md), [`ON_C15.md`](handoff/2026-10-08/ON_C15.md) |
| coverage-weighted training (rejected) | [`docs/covw_2026-10-08/README.md`](docs/covw_2026-10-08/README.md) |
| positional ADS-B join and the v35 overlays | [`docs/posjoin_2026-10-05/README.md`](docs/posjoin_2026-10-05/README.md), [`deeper_checks.md`](docs/posjoin_2026-10-05/deeper_checks.md) |
| LIRF label classes | [`docs/lirf_wrap_2026-10-06/README.md`](docs/lirf_wrap_2026-10-06/README.md) |
| train/test consistency audit; ADS-B coverage audit | [`docs/shift_2026-10-06/`](docs/shift_2026-10-06/), [`docs/adsb_audit_2026-10-07/README.md`](docs/adsb_audit_2026-10-07/README.md) |
| validation rule and protocol | [`14-validation-design.md`](docs/competitor_research_2026-09-21/results/14-validation-design.md), `docs/SOLUTION_OVERVIEW.md` §6 |
| ideas taken from other teams and how they were measured | [`public-code-deep-dive-2026-09-30.md`](docs/competitor_research_2026-09-21/results/public-code-deep-dive-2026-09-30.md), [`docs/research_2026-10-05_repos/`](docs/research_2026-10-05_repos/), [`docs/research_2026-10-05_measure/PROTOCOL.md`](docs/research_2026-10-05_measure/PROTOCOL.md) |

## 12. Licence and acknowledgements

- Code: GNU General Public License v3.0 ([`LICENSE`](LICENSE)).
- Vendored third-party code: `docs/research_2026-10-05_aobt/kpi/pylib/` holds openpyxl 3.1.5 and et_xmlfile 2.0.0.
  Both are MIT; their licence files are next to them. They are used by one research script only.
- Data: see §7 and `docs/external_data.md`.
- The challenge and its data are provided by the EUROCONTROL Performance Review Commission and the OpenSky Network.
  The ADS-B, map, airport and weather data used by the model come from adsb.lol, OpenStreetMap contributors,
  OurAirports and the Iowa Environmental Mesonet; their attribution lines are in §7.
