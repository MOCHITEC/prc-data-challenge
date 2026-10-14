# `docs/`: index

Team resilient-hedgehog, PRC Data Challenge 2026. Index written 2026-10-14.

**Start here:**
- [`FINAL_SUBMISSION.md`](FINAL_SUBMISSION.md): the final file, how its rows were produced, how to reproduce it, and
  its limits. This is the authoritative page about the final file.
- [`../README.md`](../README.md): task, model design, results (§10) and the documentation map (§11).

The folders are research notes, one per topic and date. Many are in Japanese. Most notes are pre-registrations
(`plan.md`, committed before the run) followed by results.
- A note describes the state on its own date. Later notes may overturn it.
- For the final file, `FINAL_SUBMISSION.md` overrides any note.

**The earlier `docs/README.md`.** It was a Japanese status page last edited on 2026-09-25 ("最良 v4 = 305.03"). It
is now at [`archive/README_2026-09-25_ja.md`](archive/README_2026-09-25_ja.md), unchanged byte for byte. Notes that
cite `docs/README.md` with a line number or section, for example under `competitor_research_2026-09-21/plans/`, mean
that archived file.

## Top-level documents

| file | language | what | status |
|---|---|---|---|
| `FINAL_SUBMISSION.md` | en | the final file: identity, recipe, checks, reproduction, limits | current |
| `SOLUTION_OVERVIEW.md` | en | pipeline map: each stage, why it is there, where its code and evidence are | state at v47 (2026-10-08) |
| `DEVELOPMENT_HISTORY.md` | en | every leaderboard submission, classed; rules derived from ranking-set rows | current (2026-10-14) |
| `external_data.md` | en | every external source: URL, licence, use, status, script | current |
| `features_log.md` | ja | feature log: candidates, results, lessons on measurement | running log |
| `data_profile.md` | ja | data profile | early (September) |
| `06_error_anatomy.md`, `error_anatomy_generated.md` | ja, en | error anatomy of early OOF | early |
| `guard_audit.md` | ja | guard audit by mutation tests (2026-09-14) | early |
| `CODEX_BRIEF.md` | en | brief for an independent feature-engineering effort | historical |
| `handoff.md` | ja | fact list for hand-off (2026-09-16) | superseded |
| `research_2026-10-06_handoff.md` | ja | hand-off of the measures adopted on one branch, each under 1 s in the blend | historical |
| `external_ideas_2026-10-07_note.md` | en | stamp rates: not adopted | closed |

## Final phase (2026-10-11 to 10-14)

| folder | what |
|---|---|
| `final_phase_2026-10-11/` | `RUNBOOK_final_submit.md` (upload steps), `PUBLIC_RELEASE_CHECKLIST.md`, `IMPACT.md` (what the final set changed), `NO_FINAL_UPLOAD.md`, `tracked_large_files.tsv` |
| `final_2026-10-11/` | final-set chain scripts: `adsb_layer.py` (rebuilt ADS-B layer), `apply_overlays.py`, `laneB_deltas.py`, `assemble_step.py` (later variants), configs, logs, `adsb_models/`. Its `README.md` and `adsb_layer_README.md` describe the state on 10-11 and are superseded |
| `final_phase_2026-10-13/` | status of the peaceful-babbage thread on 10-13: Feb/Jun hold-out screens of candidate changes |
| `final_phase_gate/` | LIRF copy-overlay gate scripts for a later candidate, not in the final file |

## Model and members

| folder or file | what |
|---|---|
| `competitor_research_2026-09-21/results/` | per-member evidence with pre-registration and CI verdicts: `plainy-blend.md`, `catboost-blend.md`, `catboost-retune.md`, `blend-l2.md`, `per-airport-member.md`, `optuna-retune.md`, `mlp-member.md`, `adsb-learned-residual.md`; validation design `14-validation-design.md` |
| `competitor_research_2026-09-21/` | competitor research (`SUMMARY.md`), plans `plans/P*.md` |
| `research_2026-10-05_measure/` | measurement protocol; L6 gate retention (`l6_*`), cmix (`cmix_*`), copy ideas |
| `research_2026-10-06_nmblind/`, `research_2026-10-07_ucmix/` | lane-B NM-blind blend and LIRF ucmix: plans and verdicts |
| `shift_2026-10-06/` | 2025 → 2026 train/test mismatch in `fl_nm_rate_other` (the consistency fix) |
| `batch_2026-10-06/` | the parked batch and the C10/C12/C15 submissions; version-collision incident |
| `member_stack_2026-10-06/`, `modelb_rounds_2026-10-06/`, `modelb_capacity_2026-09-30/` | stacking, model-B rounds and capacity: no gain |
| `research_2026-10-08_blend/`, `research_2026-10-08_audit/` | matched-blend re-weighting; audit of rejected levers |
| `handoff_2026-10-05/`, `handoff_2026-10-07/` | hand-offs between sessions (R6 base; request to rebuild the ADS-B layer on C12) |

## ADS-B

| folder | what |
|---|---|
| `appear_tier_2026-10-05/` | appear tier (aircraft position relative to its own stand), 12-month refit, EDDF coverage delta: `results.md`, `plan_*.md`, `final_adsb.py` |
| `months_2026-10-05/` | training the learned residual on more months of 2025 |
| `posjoin_2026-10-05/` | positional ADS-B join for unmatched departures; `deeper_checks.md` covers the v35 overlays |
| `rescat_2026-10-06/` | residual + apron category |
| `adsb_audit_2026-10-07/` | 2025 vs 2026 consistency audit of the residual stage (receiver-coverage shift) |
| `covw_2026-10-08/` | coverage-matched training weights: rejected |
| `nbr_adsb_2026-10-07/`, `nbr_clock_2026-10-06/` | neighbour ADS-B taxi features (rejected); neighbour-clock residual (parked) |
| `research_2026-10-08_gap/` | ADS-B headroom and the gap to 10th place |

## LIRF and label artefacts

| folder | what |
|---|---|
| `lirf_wrap_2026-10-06/` | LIRF "one day + taxi" class, copy band, time-zone class |
| `lirf_fill_2026-09-25/` | predicting schedule-time reuse on LIRF matched rows (pre-registration) |
| `investigation_2026-09-16/`, `data_audit_2026-09-18/` | early investigations and data audit (rollover, reparameterisation, fold shape) |
| `bigerr_2026-10-06/`, `research_2026-10-08_tails/` | large-error rows: nothing predictable / nothing adopted |

## Features, external data and other research

| folder | what |
|---|---|
| `osm_stand_2026-09-28/` | OSM stand-to-runway-end distance (pre-registration) |
| `weather_v2_2026-09-25/`, `weather_2026-10-06/` | weather repair and Open-Meteo precipitation and snow; extra weather columns: no gain |
| `opdi_2026-09-30/`, `opdi_events_2026-10-05/` | OPDI turnaround and exit-parking events (research only per `external_data.md`; EUROCONTROL terms, see the release checklist §3.1) |
| `article_fe_2026-09-24/`, `next_fe_2026-09-30/`, `research_2026-10-05_newstand/`, `research_2026-10-06_stamp/`, `research_2026-10-08_daydemean/`, `research_2026-10-08_floor/`, `research_2026-10-08_ucday/` | further feature and lever pre-registrations |
| `research_2026-10-05_drift/`, `drift_2026-10-08/`, `research_2026-10-08_winter/` | 2025/2026 drift screens: no actionable fix |
| `research_2026-10-06_corrector/`, `external_ideas_2026-10-06/`, `v28_graft_2026-10-02/` | stage-2 corrector prototype; external ideas A/B/C; v28 graft |
| `gm_research_2026-10-01/`, `research_2026-10-01/`, `research_2026-10-05_pastcomp/`, `research_2026-10-05_repos/` | surveys: top solutions, past competitions, public repositories |
| `research_2026-10-05_strict/` | stricter acceptance checks and a frozen hold-out |
| `research_2026-10-05_aobt/` | AOBT audit and the official taxi-out KPI (includes vendored openpyxl, MIT) |
| `analysis/` | multi-angle data analysis (2026-09-24) |

## Archive

| file | what |
|---|---|
| `archive/README_2026-09-25_ja.md` | the earlier `docs/README.md` (Japanese status page, 2026-09-25), unchanged |
