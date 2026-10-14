# Public release checklist (written 2026-10-14 ~02:30Z)

Repository: `https://github.com/MOCHITEC/prc-data-challenge`. It is **private**; its visibility was not changed.

State at the time of writing:

| item | value |
|---|---|
| deadline | 2026-10-14T10:00Z. The organisers review code and docs up to the last commit before it. |
| `main` | fast-forwarded from 1000e4c to the integration branch `integration/final-2026-10-14` |
| GitHub default branch | **still `claude/prc-data-challenge-planning-l7dk7k`** (b031609, 2026-09-11, old planning README). `gh api -X PATCH repos/MOCHITEC/prc-data-challenge -f default_branch=main` was refused: "Repository settings writes are not permitted through this proxy." (HTTP 403) |
| licence | `LICENSE` is the GPLv3 text; GitHub detects `GPL-3.0` |
| tests on the integration branch | 364 passed, 2 skipped (`pytest -q`; at a8ce052 and again at 0a0799b, 8 min 44 s) |
| secret scan | no hits (§4) |

## 1. Before the deadline (10:00Z)

1. **Set the default branch to `main`.** GitHub → Settings → General → Default branch → switch to `main` → Update.
   This could not be done from this session (see above).
2. **Check that the docs name the file that was actually uploaded** (done after the upload, 2026-10-14).
   - Uploaded: `resilient-hedgehog_final.parquet`, md5 **`c0d4a1439862e68852b65f6854322a29`**, 2,151,179 bytes,
     bucket `last_modified` **2026-10-14 10:30:29Z**, read back and verified. It is
     `handoff/2026-10-13/final_v5_plus_febjun_appear.parquet` from **`ea0e0a8`**, restored on `main` in `0d43558`.
     The ledger lines (`"kind": "final"`) were committed in `8189b58`. The scorer's result file (10:30:38Z) says
     `"Succeeded"`, `"used_pairs": 670790`, no score (`final_result_2026-10-14.json` in this folder).
   - `docs/FINAL_SUBMISSION.md` §1 is the one place that states the file's identity. README §3 and the runbook (header
     and §0 variables) carry the same commit and md5. Done; `grep -rn c0d4a143 README.md docs/` lists every copy.
   - The pages name **`be8d89dc…`** (`handoff/2026-10-13/final_from_v48.parquet` @ `25f6246`) only as the documented
     base: the uploaded file = that base + the 12-month appear increment on 103,311 Feb and 123,624 Jun rows
     (`docs/FINAL_SUBMISSION.md` §2.5). Until the upload these pages named `be8d89dc…` as the final file.
   - `scripts/final/assemble_final_v48.py --febjun-appear handoff/2026-10-13/feb_jun_appear_increment_on_blend5.parquet`
     rebuilds `c0d4a143…` byte for byte (`docs/FINAL_SUBMISSION.md` §5.2).
   - The pages do not name as final the earlier version `558c736` / `c7ffa31e…`, or the later files the branches
     recommend: laughing-cray's `handoff/2026-10-13/final_recommended.parquet` (`d0ce32b6…`, then v6 `7e53142e…`)
     and beautiful-davinci's `handoff/2026-10-14/final_v4_gatefull_plus1delta.parquet` (`b7e8298d…`).
     `docs/FINAL_SUBMISSION.md` §3, §4 and §7 state that those were not submitted, and why v6 was not (v50/v51).
   - **Deadline.** The organisers' pages state "2026-10-14T12:00 CET (2026-10-14T12:00+02:00)" and
     "2026-10-14T10:00Z". The upload (10:30:29Z) and the commits that record and document it (`8189b58` onwards)
     came after 10:00Z. Whether they are accepted is for the organisers.
3. **Put the final upload's ledger lines on `main` before 10:00Z.**
   - `RUNBOOK_final_submit.md` §8 commits them to `claude/model-b-and-arrival-features`, which `main` is now ahead of.
   - Merge that branch into `main` and union `experiments/submissions.jsonl` (append-only).
   - Or commit the lines on `main` directly.
4. Do not commit anything after 10:00Z that should be reviewed. (Not kept: see item 2, "Deadline".)

## 2. After the deadline

1. **Flip the visibility to public.** GitHub → Settings → General → Danger Zone → Change visibility → Public. Or:
   `gh repo edit MOCHITEC/prc-data-challenge --visibility public --accept-visibility-change-consequences`.
   - Every branch and the full history become public.
   - The secret scan (§4) covered all of them.
2. **Send the repository link** `https://github.com/MOCHITEC/prc-data-challenge` to **challenge AT opensky-network DOT
   org**, with the team name `resilient-hedgehog`. The organisers will fork it.
3. Optional: a JOAS paper is encouraged on the challenge site. `docs/research_2026-10-05_aobt/` was written with that
   in mind.

## 3. Still open (decide before or right after going public)

1. **EUROCONTROL-derived files.** These come from EUROCONTROL data whose terms say "not used for commercial purposes"
   and "may not be modified without prior written permission". The eligibility page asks for additional data "openly
   available under an open source license". Neither file is used by any submission or the final chain
   (`docs/external_data.md`).
   - `handoff/2026-10-05/opdi_exitpark.parquet` (OPDI);
   - `docs/research_2026-10-05_aobt/kpi/pru_published_2024_2025.csv` (PRU taxi-out).

   **Decided 2026-10-14 (user, option A): removed from the tip on 2026-10-14; still present in git history.**
   History was not rewritten.
   - Removed with `git rm`: the two files above and `docs/research_2026-10-05_aobt/kpi/out/validation_per_airport.csv`,
     which copies PRU columns (`pru_add`, `pru_valid_share`, …).
   - Kept: the aggregate OPDI text results and the figures.
   - How to re-create each file, and the reason (the EUROCONTROL terms, quoted): `handoff/2026-10-05/README.md` §6,
     `docs/research_2026-10-05_aobt/README.md` and `docs/external_data.md` (derived-data table).
   - README §7 carries a EUROCONTROL attribution line.
   - No code path used by the tests reads these files (`tests/test_opdi.py` mocks the OPDI table). `pytest -q` after
     the removal: 364 passed, 2 skipped (7 min 44 s).
2. **Competition-derived per-row fields. Decision pending with the user / organisers.** The files were not
   deleted or changed. Facts from a read-only audit of `origin/main` on 2026-10-14:
   - **No raw competition file** (`training_*`, `ranking`, `submitting`, `final_*`) is tracked on any ref.
   - **412 tracked parquet files** (142.8 MB, 1,410,814 rows, all 2026 rows) carry per-row fields copied or renamed
     from the competition data. Every one has `ADEP_mvt`, `t_mvt` (= `MVT_TIME_UTC_mvt`) and `t_aobt3`
     (= `AOBT_3_flt`). Subsets also carry `t_sched` (= `SCHED_TIME_UTC_mvt`), `RUNWAY_mvt` and
     `AIRCRAFT_TYPE_mvt`. The renaming is in `analysis/stand_common.py` lines 51 and 54 and
     `docs/research_2026-10-01/errgroups/ltfm_adsb/s1_ref.py` lines 409–412.

     | class | files | where |
     |---|---|---|
     | c1 | 149 | `handoff/2026-10-13/adsb_store/pos/` 62, `handoff/2026-10-13/adsb_store/posday/` 57, `handoff/2026-10-13/june_adsb_for_adsb_layer/pos/` 30 |
     | c2 | 85 | the other pos/posday files in the same three folders; they also carry `t_sched` |
     | c3 | 178 | `handoff/2026-10-13/adsb_store/feat/` 115, `handoff/2026-10-13/june_adsb_for_adsb_layer/feat/` 59, and four `learn/feat_*2026*.parquet` (`handoff/2026-10-11/adsb/learn/feat_2026.parquet`, `feat_2026fj.parquet`; `handoff/2026-10-13/adsb_store/learn/feat_2026.parquet`, `feat_pos_2026.parquet`) |

     The earlier version of this item counted only class c1 (149 files).
   - **One CSV**, `docs/research_2026-10-01/rollover/fingerprints/rank_candidates_ms8h.csv` (22 rows): `FLIGHT_mvt`,
     scheduled and movement times, `RUNWAY_mvt`.
   - **`handoff/2026-10-13/adsb_store/rescat/codes.parquet`** derives `airline` from the first 3 characters of
     `CALLSIGN_flt` / `FLIGHT_mvt`, and `apron` from `ADEP_mvt` + the `STAND_mvt` prefix.
   - **Licence.** No licence had been published for the 2026 dataset (organiser site and its source repository
     checked 2026-10-14). The 2024 and 2025 datasets were later released under CC BY 4.0.
   - `.gitignore` line 15 ("competition data (redistribution not allowed until public release)") and
     `docs/research_2026-10-05_aobt/BRIEF.md` line 12 are the team's own notes. They cite no organiser source.
   - Many other hand-off files are only `MVT_ID_mvt` + prediction or delta.
   - README §7 now carries a "Data notice" saying this and asking the organisers to contact the team if removal is
     required.
3. **ODbL notice for derived ADS-B / OSM data.** README §7 and `docs/external_data.md` state that the committed
   adsb.lol- and OSM-derived files are made available under ODbL 1.0. Confirm this is intended. ODbL requires it if a
   derived database is published.
4. **Absolute paths.** Several final-chain scripts hard-code paths of the machines they ran on (`/home/user/pb-work`,
   `/home/user/prc-final`, scratch directories). The path table at the end of `handoff/2026-10-13/README.md` maps them
   to the committed copies. The scripts were not edited.
5. **Japanese-language notes.** Many notes and hand-off READMEs are in Japanese, including `handoff/2026-10-13/README.md`,
   the main recipe. `README.md`, `docs/external_data.md`, `docs/DEVELOPMENT_HISTORY.md`, `docs/SOLUTION_OVERVIEW.md`
   and the runbook are in English.
   - Update 2026-10-14: `docs/FINAL_SUBMISSION.md` §2.2, §5.3 and §8 translate the parts of
     `handoff/2026-10-13/README.md` (as of `25f6246`) that concern the final file.
   - `docs/README.md` is now an English index. The earlier Japanese page is kept at
     `docs/archive/README_2026-09-25_ja.md`.
6. **The unknown 10-09 final upload.** The runbook's `--confirm-final-overwrite` upload replaces it. If the final
   upload did not happen, that object is still the team's final file.
7. **The earlier integration attempt** (`docs/integration_2026-10-06.md`, tip 6d3c407) is on no remote branch and
   not in the local object store. It was never pushed and is lost. Its three findings were re-checked here (§5).

## 4. Hygiene done on the integration branch

- **Secret scan**, run on 2026-10-14 with a scanner kept outside the repository. It found nothing:
  - Exact-value search for the 7 credential variables set in the container (`OSN_ID`, `OSN_ACCESS_KEY`,
    `OSN_SECRET_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GH_TOKEN`, `GITHUB_TOKEN`). It covered all 7,239
    objects of the object store, reachable or not, and the working tree: **0 hits**.
  - Patterns over `git log -p --all` (786 commits, all 11 branches on GitHub, added lines and commit messages):
    AWS key ids, private-key blocks, GitHub / Slack / Anthropic / OpenAI / Google tokens, JWTs, URLs with passwords,
    literal secret assignments. **0 hits.**
  - The same patterns over the 2,135 tracked files at the tip (0a0799b): **0 hits**.
  - Risky file names ever committed: only `.env.example`. Every version of it has empty credential values.
- **LICENSE**: GPLv3 full text, present.
- **Vendored code**: `docs/research_2026-10-05_aobt/kpi/pylib/` (openpyxl 3.1.5, et_xmlfile 2.0.0; MIT) had no
  licence files.
  - The copies are still used by `00_extract_pru.py`, so they were kept.
  - `LICENCE.rst`, `LICENCE.python` and `AUTHORS.txt` were added from the matching wheels, with a `README.md`.
  - The package files are byte-identical to the wheels.
- **Tracked `.pyc`** `docs/final_2026-10-11/__pycache__/adsb_layer.cpython-311.pyc` untracked.
- **Large files.**
  - Every tracked parquet file and every other file ≥ 1 MB is listed with its size in
    `docs/final_phase_2026-10-11/tracked_large_files.tsv`: 485 files, 423.1 MB.
  - The tip tracks 2,135 files (441.6 MB), including 477 parquet files (322.1 MB). The largest file is 30.0 MB, under
    GitHub's 100 MB limit. The packed history is 250 MB.

  | directory | files | MB |
  |---|---|---|
  | `handoff/2026-10-11/adsb/` (ADS-B features, appear, 3 residual models) | 7 | 120.8 |
  | `handoff/2026-10-13/members/` (member predictions) | 13 | 56.1 |
  | `docs/final_2026-10-11/adsb_models/` (4 LightGBM text models) | 4 | 52.4 |
  | `handoff/2026-10-13/inputs/` | 12 | 51.7 |
  | `handoff/2026-10-13/adsb_store/` (feat, learn, rescat, pos, posday) | 293 | 73.7 |
  | `handoff/2026-10-13/june_adsb_for_adsb_layer/` | 118 | 18.9 |
  | candidate and delta files in `handoff/2026-10-0{2,5,6,7,8}`, `2026-10-13`, `2026-10-14` | 37 | 48.2 |
  | `experiments/experiments.jsonl` | 1 | 1.3 |

## 5. Integration record

`integration/final-2026-10-14` was created from `origin/claude/model-b-and-arrival-features` (376ba42). Three branches
were merged into it at their tips with `git merge --no-ff`, because the final file was built with their code.

| branch | tip | merge commit | conflicts and resolution |
|---|---|---|---|
| `claude/laughing-cray-vvwh0k` | 6adbcac | 59e6b57 | `docs/posjoin_2026-10-05/deeper_checks.md`: ours was a prefix of theirs, so theirs was taken. `experiments/submissions.jsonl`: union. |
| `claude/peaceful-babbage-md9l2x` | 7fdf8f9 | baf8bc9 | `experiments/experiments.jsonl` and `pipeline.jsonl`: union. `src/` merged cleanly. |
| `claude/beautiful-davinci-ehugk4` | 50e7f63 | 862e7dc | The three `experiments/*.jsonl`: union. `docs/features_log.md`: both appended sections kept. `scripts/adsb_learn_{apply,cv}.py`: all `--base` options of both sides plus theirs' `--opdi`/`--appear` flags. `src/prc/cv.py`: see below. |

- **Append-only logs.** Each was resolved as our lines in order, then their lines not already present. Exact
  duplicate lines were dropped. The results: `submissions.jsonl` 101 lines, `experiments.jsonl` 178, `pipeline.jsonl`
  49. All are valid JSON with no duplicates. Two pairs of v42 and v47 lines differ (two different uploads under one
  number), so both lines of each pair are kept.
- **`src/prc/cv.py`, a default-behaviour change.**
  - beautiful-davinci 9ecc3bd appended the `aso` / `fmj` folds unconditionally. Its comment said they are "only run
    when a config names it", but `run_cv` uses every fold when a config has no `folds:`. So 33 configs without
    `folds:` would have gained two folds.
  - peaceful-babbage's `PRC_FEBJUN_FOLDS=1` folds are opt-in.
  - Both are kept. `aso` / `fmj` are now built only when named: `month_block_folds(..., named=cfg.folds)`, passed
    from `train.run_cv` and `scripts/cv_per_fold.py`.
  - The default fold list is the one the final pipeline used; pb-work 6a97bf0 has no `aso` / `fmj`. The two logged
    runs that use them name them in `folds:` (exp ids `main_flseq-87ea8fee1d`, `main_flseq_osm_nolirf_l2-f251478c1a`).
- **Config registry** (a8ce052).
  - Five configs were unregistered in the config-id tests: `ox.yaml`, `ox_stress.yaml`,
    `main_flseq_osm_nolirf_l2_clip.yaml`, `main_flseq_asofmj.yaml`, `main_flseq_osm_nolirf_l2_asofmj.yaml`.
  - They were added to the five `BEFORE` registries with ids computed from the unchanged files. The ids equal the
    logged exp ids.
  - `test_mlp_member.py` also asserted that every registered recipe is non-MLP, which fails for peaceful-babbage's
    registered `main_flseq_osm_nolirf_l2_mlp_l6.yaml`. That check now skips `*_mlp*` names; the id check stays.
- **Not merged:**
  - `codex/feature-engineering` (8bdf4f9, 2026-09-19). It conflicts in `src/prc/features.py` and
    `tests/test_features.py`. Its one surviving feature, `stand_arrival_gap`, was landed separately in this lineage
    (4d41e60). The final chain does not use the branch.
  - `claude/quirky-hopper-b7q73j` (f92e3ef, 2026-09-28). It conflicts in `src/prc/config.py`, `src/prc/io_minio.py`,
    `sql/000_admin_manual.sql` and `.env.example`, and has modify/delete conflicts on `docs/05_runbook.md` and
    `docs/decisions.md`. It holds a Snowflake presigned-URL load path; the final chain does not use it.
  - Both branches stay on GitHub as they are.
- **Already contained** in the integration branch, so nothing to merge: `main`, `claude/clever-bell-jz8lut`,
  `claude/prc-data-challenge-phase-1-5kqx25`, and `claude/prc-data-challenge-planning-l7dk7k` (the current default
  branch).
