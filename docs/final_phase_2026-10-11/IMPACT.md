# Final phase: what changed, and what it means for our pipeline (2026-10-11, ~15:00 UTC)

This is a read-only investigation. Nothing was trained, submitted or uploaded, and nothing in `src/` or `configs/` was
edited. Facts carry a source: a URL, a commit, a file path, or a command run here. Inferences are marked **[X]**.
Statements that come from another session's notes and were not re-measured here are marked **[N]**.

## 0. Corrections to the brief

The brief was the user's relay of the organisers' Discord messages. These facts differ from it or add to it.

- **The board best is v48 = 234.4099, not v42.** v48 was processed on 2026-10-11 at 14:47:38Z (leaderboard API). After
  it came **v49 = 248.6458**, processed at 14:57:10Z. That was a diagnostic run: §3.3 and §5.
- **The data page names the template `final_submitting.parquet`**, not `final_submission.parquet` (§1.2).
- **Final-phase work has already started**, on `claude/beautiful-davinci-ehugk4`: b53bdaa, 4ce4a8c and 169e7f4, all
  2026-10-11 between 14:47Z and 14:57Z (§3.1).
- **A `resilient-hedgehog_final.parquet` is already in the team bucket.** It was uploaded on "10-09 16:02"; the note
  gives no time zone. The organiser's scorer marked it Failed, because of an error on their side. No ledger or commit
  on any branch records who made it [N: `handoff/2026-10-11/FINAL_PHASE.md` @ b53bdaa].
- **The competition data could not be checked from this container.**
  - No OSN credentials are set: the `OSN_*` variables are unset and there is no `.env`.
  - `.venv` does not exist either. A scratch venv (polars and pyarrow only) was used to read committed files.

---

## 1. Official website (fetched 2026-10-11 ~14:41Z)

### 1.1 Where the pages are

- **Base URL.** `https://prc-data-challenge-2026.netlify.app/`, as referenced in `README.md` and `refs/`. No
  `/tmp/prc_*.qmd` exists in this container.
- **Pages that exist.** `index.html`, `data.html`, `ranking.html`, `eligibility.html` (the "Terms & Conditions") and
  `rationale.html`, all HTTP 200.
- **Pages that do not exist.** `rules.html`, `faq.html`, `submission.html`, `about.html` and `sitemap.xml` all
  return 404. There is no separate rules or FAQ page.
- **Page source.** The site's source is also at `raw.githubusercontent.com/euctrl-pru/prc_data_challenge_website_2026/main/*.qmd`.
  - Its `index.qmd` differs from the deployed page (see deadlines).
  - The GitHub API for that repo is blocked for this session, so its commit dates are unknown.

### 1.2 What the pages say now (verbatim)

**Final phase and file names**
- data.html, "The Final Datasets": *"Same as the ranking and submitting dataset but covering two additional months,
  February amd June 2026: final_ranking.parquet, final_submitting.parquet"*.
- ranking.html, "Final Submission": *"Please submit you unique final submission using the following filename:
  `<team-name>_final.parquet`"*.
- ranking.html, "Submission Instructions": *"Teams will be ranked based on the best Root Mean Square Error (RMSE)
  achieved across their submissions."*, and *"There is currently a limit of 5 submissions per day (and a total max of
  1GB per bucket.)"*
- index.html, "Ranking", not updated: *"The applied ranking will use RMSE for submitted predictions for the movements
  in January and July 2026."*

**Deadlines**
- Deployed index.html: *"The competition runs from the 1st of September 2026 up until the 14th of October 2026
  12:00:00 CET."* and *"Deadline is extended till 2026-10-14T12:00 CET (2026-10-14T10:00Z)"*.
- GitHub `index.qmd` (raw, fetched 14:42Z): *"up until the 11th of October 2026 23:59:59 CET"* and *"We are having a
  final submission for which the deadline is 2026-10-14T12:00 CET (2026-10-14T12:00+02:00)"*.
- eligibility.html (deployed and GitHub alike): *"Final single submission is uploaded by 2026-10-14T12:00 CET
  (2026-10-14T12:00+02:00)"*.
- Both explicit offsets give **2026-10-14T10:00Z**. [X] Literal CET (+01:00) would be 11:00Z, but every explicit
  offset on the site is +02:00 / 10:00Z.

**Whether the last upload counts.** The site says nothing about re-uploads, overwrites or which upload counts. The
only wording is "unique final submission" and "Final single submission is uploaded by …".

**Repo and docs requirements (eligibility.html, "Eligibility for Prize")**
- *"All used external datasets are openly accessible/usable and documented."*
- *"All produced source code are made openly available on GitHub under the GNU GPLv3 license. Note: It will then be
  forked by the Challenge GitHub account for organisational purposes."*
- *"All additional datasets used are openly available under an open source license."*
- *"Sufficient documentation is provided to understand and reproduce the results."*
- *"The solution is original: … simply re-using existing code and rewriting the data input and output mechanism is
  not sufficient …"*
- index.html adds: *"openly sharing final solutions (incl. code and documentation on a public GitHub repo)"*. It also
  encourages a JOAS paper.

**Review criteria.** No page states review criteria. The only related text is the ranking.html note: *"We will monitor
submissions for attempts to learn from or exploit the ranking process. We consider such practices unfair and
inconsistent with the purposes and goals of the Data Challenge."*

**Not on the site** (Discord only, as relayed):
- the three-set ranking;
- the review of the top 30;
- the ±12 s dithering;
- the e-mail address for the repo link;
- "design should not change";
- the AOBT_3/LOBT question;
- EUROCONTROL aggregates.

**Team page.** `teams/resilient-hedgehog.html`: Industry, Japan, 1 member, MOCHITEC Inc.

---

## 2. Data

### 2.1 Bucket and access

- **Bucket name.** `prc-2026-datasets`. It is not in `configs/`. Commit f92e3ef on `quirky-hopper` says: *"The
  default competition bucket was a name that does not exist; it is now prc-2026-datasets"*. On the working branches,
  `.env.example` leaves `OSN_COMPETITION_BUCKET=` empty.
- **Listing.** `io_minio.list_objects` could not run: there are no credentials in this container.
- **Anonymous probes.** A HEAD request on `s3.opensky-network.org/prc-2026-datasets/{ranking, final_ranking,
  final_submitting, nonexistent_xyz}.parquet` returns **403 on all four**, and the bucket listing also returns 403.
  Existence therefore cannot be told apart from absence.
- **What was not measured here:**
  - row counts;
  - date range per month;
  - airports;
  - column differences;
  - the Jan/Jul dithering share and size;
  - whether the template ids match.

### 2.2 What another session recorded [N: `handoff/2026-10-11/FINAL_PHASE.md` @ b53bdaa]

- *"The organisers published `final_ranking.parquet` / `final_submitting.parquet` at 2026-10-08 11:04Z (bucket
  prc-2026-datasets)."*
- *"670,790 rows for Jan, Feb, Jun and Jul 2026. They include the 344,841 rows of the current LB (Jan and Jul); the
  325,949 rows of Feb and Jun are new. The columns are the same as in ranking."*
- That note does not quantify the ±12 s take-off-time differences.

**The two new months are February 2026 and June 2026** (data.html).

---

## 3. Latest repo state across all branches (fetched 2026-10-11 14:40Z and ~14:58Z)

### 3.1 Final-phase work

These searches were run:
- `git grep` for `final_ranking|final_submitting|final_submission.parquet|_final.parquet|final phase` on every
  remote branch tip;
- `git log --all -S"final_ranking"` and `-S"_final.parquet"`.

Only **`claude/beautiful-davinci-ehugk4`** (session `01QkQB6W…`) has 2026 final-phase work:

| commit | time (UTC) | what |
|---|---|---|
| b53bdaa | 2026-10-11 14:47:43 | v48 logged; `handoff/2026-10-11/FINAL_PHASE.md`; `scripts/adsb_stage1_feasibility.our_departures` reads `data/raw/final_ranking.parquet` for 2026 months 02/06 only (Jan/Jul still from `ranking.parquet`); "Feb/Jun 2026 ADS-B ingest started" (58 days, "3 to 4 hours" expected) |
| 4ce4a8c | 2026-10-11 14:48:09 | `docs/appear_tier_2026-10-05/build_features.py`: adds the `2026fj` day set (Feb 28 + Jun 30 days) |
| 169e7f4 | 2026-10-11 14:57:23 | `docs/appear_tier_2026-10-05/final_adsb.py` (ADS-B residual on any pre-ADS-B base, for Jan/Feb/Jun/Jul rows); v49 logged; note: "the base of the bucket's final.parquet is not usable" |

All other hits, on every branch, refer to the 2024/2025 editions:
- `docs/README.md:29`;
- `docs/handoff.md:25`;
- `docs/investigation_2026-09-16/hypotheses/external.md:9`;
- `docs/03_prior_art.md`;
- the competitor-research write-ups.

### 3.2 Branch tips

| branch | tip | date (UTC) |
|---|---|---|
| beautiful-davinci-ehugk4 | 169e7f4 | 2026-10-11 14:57 |
| laughing-cray-vvwh0k | 49b9d98 | 2026-10-08 06:47 |
| peaceful-babbage-md9l2x | 80a0ac3 | 2026-10-08 06:44 |
| model-b-and-arrival-features | 2f597ff | 2026-10-07 05:35 |
| clever-bell-jz8lut | 1fdaca7 | 2026-09-28 |
| quirky-hopper-b7q73j | f92e3ef | 2026-09-28 |
| main | 1000e4c | 2026-09-25 |
| codex/feature-engineering | 8bdf4f9 | 2026-09-19 |
| prc-data-challenge-phase-1-5kqx25 | d4fd126 | 2026-09-17 |
| prc-data-challenge-planning-l7dk7k (**GitHub default branch**) | b031609 | 2026-09-11 |

- laughing-cray, peaceful-babbage, beautiful-davinci and model-b diverged at 632bc96 (2026-10-05). None of them
  contains the others.
- `experiments/submissions.jsonl` is split across branches: each branch lacks other sessions' entries. For example,
  v48 and v49 exist only on beautiful-davinci.

### 3.3 Board (leaderboard API, team filter, 48 items; full board 5,363 items / 242 teams)

| ver | score | processed (UTC) | content | made by |
|---|---|---|---|---|
| v42 | 236.4369 | 10-07 02:06 | C12 = v40 + C10 (fl_/seq_ fix, LIRF time-zone class, copy band) + apron residual swap + L6/cmix member deltas + NM-blind model-B blend + L6 in the MLP member, then LOBT clip | laughing-cray 0b16e18 |
| v43 / v44 / v45 | 238.6626 / 237.4070 / 238.3508 | 10-07 03:29 | isolation runs on v40: apron only / C10 only / peaceful-babbage member deltas only | laughing-cray ace5f58 |
| v46 | 236.4631 | 10-07 04:20 | C12 + 12-month appear delta | model-b fd56735 |
| v47 | 234.4831 | 10-08 01:28 | **C15** = v42 + per-member fl_/seq_ fix + 3-seed L6 deltas + ucmix (356 LIRF unmatched rows) + LOBT clip | laughing-cray e0fb04d |
| **v48** | **234.4099** | 10-11 14:47 | C15 + 12-month appear delta ("A") + EDDF coverage-matched residual delta. File: `handoff/2026-10-08/C15_appear12_cov26eddf.parquet` | beautiful-davinci b53bdaa |
| v49 | 248.6458 | 10-11 14:57 | "Final-phase fallback check": the Jan/Jul rows of the 10-09 bucket `final.parquet` + 12-month appear residual | beautiful-davinci 169e7f4 |

- **Rank.** Team best 234.4099 is **rank 23 of 242** by best score. 30th place is 237.1154; 1st is 212.8457. v47 was
  rank 20/237 when scored [N: `docs/batch_2026-10-06/README.md`].
- **Version collisions** [N: `docs/batch_2026-10-06/README.md` @ cf0f289; 33327e5]. In both cases the later upload
  replaced the earlier object and the board entry was re-scored:
  - v42: model-b's upload (10-06 15:00, "v40 + R6/cbopt delta", 660ca56) was overwritten by laughing-cray's C12
    (10-07 02:06).
  - v47: beautiful-davinci's "A" (01:21, 236.3563) was overwritten by laughing-cray's C15 (01:28).
- **The C15 chain**, one delta layer per submission:

```
v33  blend + learned ADS-B residual
v34  + 430-row positional overlay
v35  MLP 4-member blend (other session)
v36  + posjoin learned residual, LIRF copy blend, LOBT clip
v37  + 4-month residual
v38  + V2 residual settings
v39  + appear tier
v40  + 7- and 12-month residual swaps
v42  C12
v47  C15
```

  Sources: `docs/SOLUTION_OVERVIEW.md` (eca0bc8), `docs/months_2026-10-05/README.md`, `docs/posjoin_2026-10-05/*`.

---

## 4. Impact on our pipeline: producing a 4-month final

### (a) ADS-B

- **2026 days covered today.** Only **Jan and Jul 2026, 62 days**.
  - `scripts/adsb_ingest_all.py::all_days()` hard-codes months `(1, 7)` of 2026. A `--days` override exists.
  - Every 2026 ADS-B product derives from those 62 days:
    - the posjoin files `$SCRATCH/pos/v2026.*_pos.parquet` and `dwell_2026.parquet`;
    - the appear "2026 ADS-B stand map";
    - `feat_2026` / `appear_fill_2026`.
- **Training data of the residual models** (2025 only) [N: `docs/months_2026-10-05/README.md`, `docs/SOLUTION_OVERVIEW.md` §3]:
  - all 12 months of 2025 ("7 months at LFPG");
  - LIRF excluded from the swaps;
  - c = 0 at LTFM, where there is no ground coverage.
- **A new adsb.lol ingest of Feb and Jun 2026 (58 days) is needed.** None of the files above exists for those months.
  - beautiful-davinci started it at about 14:47Z on 10-11 (b53bdaa).
  - Before b53bdaa, `our_departures` read only `ranking.parquet` for 2026.
- **Throughput figures:**
  - 2026 Jan/Jul ingest: 57 days downloaded, **194.9 GB, 3 h 47 min wall, 2.5 to 3.5 min per day**, 2.20 to 4.28 GB
    per day (`adsb-full-ingest-candidate.md`).
  - 2025 Jan/Jul labelled ingest: **166.11 GB streamed, 141 min of steps plus about 2.8 h of memory-gate waits**,
    1.68 to 3.85 GB per day (`docs/competitor_research_2026-09-21/results/adsb-ingest-2025-labelled.md` §3).
  - [X] At the same rates, 58 days come to roughly 180 to 210 GB and 2.5 to 6 h. Ground coverage for Feb/Jun 2026 is
    unknown: `docs/adsb_audit_2026-10-07` found a receiver-coverage shift between months.

### (b) Row-specific overlays and hand-tuned steps in the C15/v48 chain

| step | how rows are chosen | generalises to Feb/Jun by code? | reproducible from repo? |
|---|---|---|---|
| **430-row positional overlay** (`analysis/local_posjoin_overlay.py`, 74efe7d; list in `docs/posjoin_2026-10-05/replacements_v33.csv`, 745cef9) | Rule, not an ID list: AOBT_3 null, not LIRF, push ≥ 0.5, no later stop ≥ 900 s (w = 1); long-stop rows at `W_LONG = 0.75`. The rule was designed on the EHAM 2026-01-02..07 snow days in the ranking set (README: "v33 predicts about 2,400–2,500 s… ADS-B ground tracks give 3,350–3,750 s"); w was chosen on 2025 noise | Yes, if `pos/v2026.*_pos.parquet` and `dwell_2026.parquet` are built for Feb/Jun (the glob would pick them up) | The script is in git; its inputs live in `$SCRATCH` (not in git). The CSV is in git but only covers the 430 Jan/Jul ids |
| **LIRF label classes** (`analysis/local_lirf_artefacts_apply.py`, 48db3bb) | Thresholds on MVT−SCHED with q rates from 2025. C = copy band, 7–13.5 h → MVT−SCHED. B = time-zone class, per-prefix q. C7r = one-day wrap. The C7r q values were revised per 2026 row; the docstring names the flights `TAY9LA` and `NOSOS431` | By code, but line 38 hard-codes `data/raw/ranking.parquet` | **Unclear.** The committed script always applies C7r as well. The notes say C7r was dropped after v41 and C10 kept only B and C, but no committed code shows how C10 left C7r out |
| **LOBT clip** (`analysis/local_lobt_clip.py`, 4cd6e9c; earlier `v12-lobt-bound`) | Identity \|y − (MVT−LOBT)\| ≤ 3606 s on all matched 2025 rows; clips 17–34 rows per build | Yes. It reads `data/work_flseq/feat_rank.parquet`, which must be rebuilt from `final_ranking` | Code in git; the feature cache is not |
| **appear tier** (`docs/appear_tier_2026-10-05/*.py`, beautiful-davinci only) | Residual delta c_A(b) − c_P(b) on rows with appear features; the 2026 stand map comes from 2026 ADS-B | Yes after the Feb/Jun ingest (4ce4a8c, 169e7f4 in progress) | The per-row deltas for C12/C15 are committed parquet keyed to Jan/Jul ids (`handoff/2026-10-0{6,7,8}/*delta*.parquet`). Model files `final_model_a2_delta_m12_v2_A*.txt` are in `$SCRATCH` |
| **Residual swaps 4→7→12 months** (`analysis/local_adsb_apply_swap.py`, 284ed6c) | file + c_new(b) − c_old(b) on ground-fix rows; LIRF excluded; LFPG kept at 7 months, explicitly "post hoc" (cb0649d) | The rule is generic, but it needs b. **b is recovered from the previous submission file by a fixed point**, and no previous file exists for Feb/Jun rows | Swap models are in `$SCRATCH`. b recovery is "ill-posed (3 plain steps leave RMS 15 s of v38 unexplained)" (466295a) |
| **ucmix / NM-blind / L6 / cmix** | Lane-level models (LIRF unmatched rows with 0 < MVT−SCHED < 8 h; unmatched non-LIRF rows at weight 0.25) | By code | Configs and `fe_retention`/`cmix` src exist **only on peaceful-babbage**. `analysis/local_nmb_laneB.py` runs "from /home/user/pb-work" (a local checkout of the peaceful-babbage code) |

### (c) Reproducibility of C15 / v48

[X] **No.** C15 and v48 cannot be regenerated end-to-end from code committed to any single branch. The facts behind
this:

- **No branch has all the components.**

  | component | branch that has it |
  |---|---|
  | the `analysis/local_*` chain | laughing-cray |
  | L6/cmix/ucmix/nmb configs and `fe_retention` src | peaceful-babbage |
  | appear tier and `final_adsb.py` | beautiful-davinci |
  | R6/cbopt | model-b |

  `main` (1000e4c) and the GitHub default branch (b031609) have none of them.
- **There is no committed assembly script for C11/C12/C15.** The commits that built and submitted them (f29efdf,
  0b16e18, e0fb04d) touch only `docs/batch_2026-10-06/README.md`, the ledger and the single-part scripts.
- **Files that exist only on other containers' disks** (git-ignored or scratch), per the notes:
  1. The `outputs/*.parquet` full-fit member predictions (a2o, pao, p2o, c2, m4, u and the L6 variants)
     (`docs/handoff_2026-10-05/R6_for_v38_chain.md` §2: "It is not in git").
  2. Every chain file from v33 to v40, plus `outputs/c10_v40_flfix_tz_copy.parquet` ("(local)") and
     `outputs/c15_v42_flfixm_seeds_ucmix.parquet`.
  3. `$SCRATCH` ADS-B data:
     - filtered fixes and joined files: about 350 days of 2025 and 62 days of 2026 in laughing-cray's container. The
       model-b container had "only January and July 2025" (`docs/handoff_2026-10-07/REQUEST_rebase_on_C12.md`);
     - `pos/*`, `dwell_2026.parquet`, `agents/audit/feat_rank_extra_union.parquet`;
     - the residual models `final_model*.txt`.
  4. Feature caches `data/work*/feat_*.parquet`, the OSM caches `data/external/osm/` (fetched 2026-09-29/30; "a later
     fetch can give a slightly different stand table") and the METAR caches.
  5. The aso/fmj fold changes used for the residual bases. They exist only as diffs
     (`docs/months_2026-10-05/local_patches/*.diff`) and are not applied in `src/`.
- **Base members are not persisted.** `src/prc/train.py` deletes each booster after predicting, and `src/` has no
  LightGBM/CatBoost `save_model`. Only two things are saved, both locally: the ADS-B residual scripts write `.txt`
  models to scratch, and the MLP can checkpoint to `$PRC_MLP_CKPT` (`src/prc/mlp.py`). [X] So predicting Feb/Jun
  rows needs the full fits re-run for every member, which means retraining. beautiful-davinci (169e7f4) writes:
  *"This can only be done with the integrator's trained models."*
- `src/prc/pipeline.load_raw` uses `rglob("ranking.parquet")` / `rglob("submitting.parquet")`, which are exact names.
  [X] It would not pick up the `final_*` files without a code change.

### (d) The ±12 s dithering on Jan/Jul rows

Not measured here (no data). These parts of the pipeline use second-level times:

- **MVT-anchored clocks** (`src/prc/features.py`): `mvt_minus_aobt3` (the anchored proxy), `mvt_minus_lobt`,
  `mvt_minus_sched`, `mvt_minus_eobt1`, `mvt_minus_iobt`.
  - `aobt3_minus_eobt1`, `aobt3_minus_sched` and `lobt_minus_eobt1` do not involve MVT.
- **Runway-sequence features** (`src/prc/features_extra.py`, all listed in the production `include_candidates`):
  - `seq_hw_prev_pairq`, `seq_hw_excess`, `seq_hw_mean_prev5/next5`, `seq_hw_min_prev3`;
  - `seq_burst_*`, `seq_arr_prev_s/next_s/since_prevdep/pm10m`, `seq_rwy_share_pm30`;
  - `dup_mvt_span`.
- **Neighbour windows**, base features per `docs/SOLUTION_OVERVIEW.md` §2: the `nbr_*` family, e.g.
  `nbr_mvt_minus_aobt_mean_60m`.
- **Arrival-side features**, which move with the dither (the organiser said arrival block and taxi times move):
  `stand_arrival_gap` (arrival `BLOCK_TIME` asof-joined to departure MVT), `occ_*`, `ta_*`, `arr_taxi_in_mean_*`.
- **ADS-B joins:**
  - posjoin: take-off within ±90 s of MVT after an airport offset (p10–p90 within ±40 s);
  - callsign join: take-off nearest MVT within ±30 min;
  - residual features relative to MVT or AOBT_3 (`res_t_move_lo`, `t_appear_adj_m_aobt3`).
  - b53bdaa reads Jan/Jul departures from `ranking.parquet`, not `final_ranking`.
- **LIRF rules.** Copy rows are defined by |y − (MVT−SCHED)| ≤ 12 s (training side); the wrap and time-zone bands use
  MVT−SCHED.
- **Off by default.** `proxy_sec_in_min` (seconds part of MVT−AOBT_3) exists but is not in the production
  `include_candidates`.
- **Time-stamp grids.** 5 s at EDDM, EGLL, LFPG, LIRF, LTFM and 1 s elsewhere [N: `docs/posjoin_2026-10-05/deeper_checks.md` #13].

[X] What follows from this:
- The anchored component y − proxy = AOBT_3 − BLOCK, and the LOBT identity term y − (MVT−LOBT) = LOBT − BLOCK. Neither
  depends on MVT.
- So **if the hidden Jan/Jul labels moved with MVT**, re-deriving the features from `final_ranking` tracks the dither.
  Reusing v48's Jan/Jul predictions unchanged (as FINAL_PHASE.md proposes) would add an error equal to the dither:
  - about +48 s² (≈ +0.10 s RMSE) if the dither is uniform within ±12 s;
  - about +144 s² (≈ +0.31 s) if its sd is 12 s.
- **If the labels did not move**, the opposite holds.
- The organiser's message settles neither case for departures.

### (e) Dependence on AOBT_3_flt / LOBT_flt

**Official definitions** (data.html; `docs/data.qmd` does not exist in this repo):
- *"LOBT_flt: last known off-block time"*
- *"AOBT_3_flt: the Actual Off-Block Time (AOBT) for flown (M3) trajectory."*

**How central these columns are:**
- **Lane split.** Matched lane = AOBT_3 non-null: *"Lane = `mvt_minus_aobt3` non-null"* (R6 hand-off §2). That is
  98.47 % of ranking departures (`docs/data_profile.md`: AOBT_3_flt and LOBT_flt null on 0.0153).
- **Anchored target.** *"The base models predict y − proxy, where proxy = MVT − AOBT_3"* (`docs/SOLUTION_OVERVIEW.md`
  §1); `docs/handoff.md` §2: `TAXITIME = proxy + g(x)`. The members a2o and pao are anchored; p2o is plain-target.
- **LOBT:**
  - the `lobt_bound: true` model option in production configs (e.g. `configs/main_flseq_osm_nolirf_l2_opt.yaml`);
  - the post-hoc `analysis/local_lobt_clip.py`;
  - features `mvt_minus_lobt`, `lobt_minus_eobt1`, `lobt_null`.
- **Rules keyed on AOBT_3 null:** the posjoin overlay (`t_aobt3.is_null()`), the LIRF artefact rules
  (`AOBT_3_flt.is_null()`), ucmix and NM-blind (unmatched lane).
- **ADS-B.** The callsign join uses `CALLSIGN_flt`, which on the scored set is "present exactly when `AOBT_3_flt` is"
  (`adsb-full-ingest-candidate.md`).
- **External evidence.** kind-mango *"dropped its exclusion of `AOBT_3_flt` on 2026-09-26. The switch took it from
  281.87 to 264.34 (−17.5 s). Every public repo above rank 100 now reads `AOBT_3_flt`"*
  (`docs/competitor_research_2026-09-21/results/25-top-teams-update-2026-09-29.md`).

[X] A ruling against these columns would remove the target anchor of the main members and the LOBT bound and clip. It
would not be a feature drop; the model would have to be redesigned, which conflicts with "the design of the model
should not change". The question is still open.

### (f) Licence and repo readiness

**LICENSE and README**
- **LICENSE.** GPLv3 full text (34,674 bytes; blob f6cdd22a, identical on all working branches and `main`).
  `gh api` reports `license: GPL-3.0`.
- **README.md** (blob 3e9752c8 on all working branches) is stale:
  - *"Status: Planning phase"*;
  - it lists `docs/01_competition_facts.md` … `04_master_plan.md`, which exist only on the planning and quirky-hopper
    branches;
  - *"Reproduction: To be written with the pipeline"*.

**`docs/external_data.md`** (blob edb4a237, identical on the four working branches)

Covered:
- adsb.lol (ODbL 1.0, with attribution);
- OSM (ODbL 1.0);
- OurAirports (public domain);
- IEM METAR (public domain; fetch range 2025-01-01 to 2026-08-01, which already covers Feb and Jun 2026).

Gaps:
- **adsb.lol is out of date.** "Days used" lists 4 labelled 2025 days + 62 scored days, and "How it enters the
  prediction" describes the v17–v20 rule override. The production use is different: a learned residual on all 12
  months of 2025, plus posjoin, appear and stand map, plus the Feb/Jun 2026 ingest now running.
- **Open-Meteo is missing.** It is CC BY 4.0 (`src/prc/weather.py:283-288`) and used only by `configs/wx2_C.yaml`,
  not by production configs.
- **EUROCONTROL OPDI is missing.** Code is on peaceful-babbage and beautiful-davinci (`src/prc/opdi.py`,
  `scripts/opdi_events_ingest.py`) and says *"LICENCE FLAG: … non-commercial, no modification"*. Commit 6c4295f:
  *"pending the licence decision"*. `docs/SOLUTION_OVERVIEW.md` (the v47 pipeline map) does not list OPDI.
- **EUROCONTROL PRU taxi-out xlsx is missing.** It is used in research only
  (`docs/research_2026-10-05_aobt/kpi/00_extract_pru.py`, peaceful-babbage).

**Other repo contents**
- **Vendored library.** peaceful-babbage (7ecd55e) vendors **openpyxl + et_xmlfile (193 files)** under
  `docs/research_2026-10-05_aobt/kpi/pylib/`, with no licence file. [X] Both are MIT, whose licence requires the
  notice.
- **Committed prediction files** (no raw competition columns; schemas checked here):
  - beautiful-davinci has **23 parquet** files under `handoff/2026-10-0{2,5,6,7,8}/`, holding `MVT_ID_mvt` + predictions,
    deltas or OOF. They are force-added despite `.gitignore: *.parquet`;
  - laughing-cray and model-b have `docs/posjoin_2026-10-05/replacements_v33.csv` (430 ids and values);
  - peaceful-babbage has 49 more CSVs (research outputs).

**Secrets.** I scanned `git log -p --all` over all 11 refs for:
- AWS key ids;
- private-key blocks;
- GitHub, Slack, Anthropic, OpenAI and Google tokens;
- literal OSN/AWS/Snowflake key assignments;
- literal MinIO constructor keys;
- 40-character mixed-case tokens.

Result: **0 hits.** The `Minio(...)` calls in `sql/015_osn_pull.sql` (94477c8), `src/prc/doctor.py` and
`docs/decisions.md` (a107b2d) pass variables. No `.env`, `.pem`, `.p8`, `.key`, `connections.toml` or `secrets.toml`
was ever committed.

**Visibility.** `gh api repos/MOCHITEC/prc-data-challenge` returns:
- `"private": true`, `"visibility": "private"`;
- `"default_branch": "claude/prc-data-challenge-planning-l7dk7k"` (tip b031609, 2026-09-11);
- `pushed_at 2026-10-11T14:48:12Z`.

### (g) Submission tooling

- **No `_final` support anywhere.**
  - `io_minio.submission_object_name()` returns `f"{team}_v{version}.parquet"`.
  - `upload_submission(local_file, version: int, …)` and `submit.submit(..., version: int, ...)` take an integer
    version.
  - `leaderboard.version_of` matches `_v(\d+)`.
- **Row-count gate.** `EXPECTED_SUBMISSION_ROWS = 344_841` (`src/prc/config.py:16`). In `validate()` a template-row
  mismatch is only a **warning**, so a 670,790-row file checked against `final_submitting.parquet` would pass the gate
  with a warning.
- **Overwrite guard.**
  - The guard from 124861b exists **only on model-b** (io_minio blob a1b79df9).
  - quirky-hopper and planning have an earlier `FileExistsError` guard.
  - laughing-cray, beautiful-davinci and peaceful-babbage (io_minio blob aaca47cb) have **no overwrite guard**.
  - [X] Extended to `_final` as written, the guard would refuse a second upload unless `overwrite=True`. Per b53bdaa,
    `resilient-hedgehog_final.parquet` already exists, so **every final upload from now on is an overwrite**.
  - The team's own observation (io_minio docstring): *"the bucket has no versioning and the scorer re-reads a re-PUT
    name"*.
- **The website** does not say that the last upload counts (§1.2). Whether `_final` uploads count against the 5/day
  limit is not stated.

---

## 5. What a reviewer may read as "optimizing for the ranking" (facts only)

Sources: `experiments/submissions.jsonl` (merged over all branches), commit messages, and notes. The site warns:
*"We will monitor submissions for attempts to learn from or exploit the ranking process."*

**Board probes and A/B submissions**, in the notes' own words:

| version | how the notes describe it |
|---|---|
| v5 | "submitted to get a third CV<->LB pair" |
| v8 | "Submitted to find out whether the code itself regressed" |
| v9 | "Diagnostic probe … Splits the 32 s gap … into matched and unmatched halves, **which offline analysis cannot do without labels**" |
| v10 | "probe: v4's 339,551 matched predictions kept byte-identical, only the 5,290 unmatched rows replaced" |
| v11 | "probe on v10: identical on 344,830 rows, **11 rows replaced** … floor −8.38 s if none of the 11 is a rollover; break-even at pi = 0.19" |
| v13 | "Matched-lane probe" |
| v19 | Commit 4bb056d "(board probe)"; ledger: "OOF verdict REJECT (LIRF +9.65 s); submitted because the board scores the full final set" |
| v20 | Not labelled a probe. It is "Lane hybrid H2", built in d912c37 "(post hoc)" |
| v25, v26, v27 | "LB A/B" |
| v30, v31 | "graft" and "Decomposition probe" |
| v43–v45 | "isolation runs", sent after a misread board value. The notes: "the isolation submissions were not needed. They still give the attribution" |
| v49 | "Final-phase fallback check": the Jan/Jul rows of a final-file candidate scored on the board (2026-10-11) |

The team's own rule (`docs/competitor_research_2026-09-21/results/14-validation-design.md` A9): *"any submission
whose predictions are chosen to make the returned RMSE informative about the labels is a label measurement and should
not be sent."*

**Board scores read back into decisions**
- **v41 back-solve.** *"Back-solving the score: about 1 of the 3 high-q one-day rows was a wrap"*. The one-day class
  was then dropped (`docs/lirf_wrap_2026-10-06/README.md`, 0d4b1db).
- **Posjoin rule kept because of the board.** *"the board gain of v34 (2,090 MSE) exceeded the estimate, which says
  the rule worked there"* (`docs/posjoin_2026-10-05/deeper_checks.md`).
- **Board-transfer ratios.** These were measured and recorded: consistency fixes "about 2× their CV estimate"; ADS-B
  add-ons "10–35 %". The v42 attribution came from v43–v45 (b6c627e).

**Row-level handling designed by looking at specific ranking rows**
- The 430-row posjoin overlay, built around the EHAM 2026-01-02..07 snow days in the ranking set.
- LIRF q values revised per 2026 row, with `TAY9LA` and `NOSOS431` named in the code (48db3bb). The C7 analysis says
  *"v40 predicts all six 2026 rows of this kind as schedule copies"*.
- LOBT clip: 17–34 rows per build.
- Time-zone class: 6 rows. Copy band: 17 rows. ucmix: 356 rows.
- Per-row delta files keyed to ranking ids are committed (beautiful-davinci `handoff/`).

**Choices labelled "post hoc" in the notes**
- LFPG kept on the 7-month residual, "post hoc" (cb0649d).
- Lane hybrid H2 (d912c37), shipped as v20.
- OSM nulled at LIRF (221779d).
- 12-month appear candidate "B (full, post hoc)" (1f20061).

**Process facts**
- Two version collisions overwrote scored objects in place (v42, v47).
- An unrecorded `_final.parquet` upload exists (10-09).
- `docs/SOLUTION_OVERVIEW.md` §6 adds: "Few-row, heavy-tailed gains are rejected after v41."
