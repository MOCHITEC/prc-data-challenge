# Development history: leaderboard use and ranking-set inspection

This file states plainly how the team used the leaderboard, and which rules were derived by looking at rows of the
ranking set.
- Every submission's purpose is taken from `experiments/submissions.jsonl`, in the ledger's own words.
- Scores come from the organisers' leaderboard API, read on 2026-10-14. The board shows the last object scored under
  each file name.
- Other statements cite a commit or a file. The leaderboard set is January and July 2026 (344,841 departures).
- The organisers' ranking page says: "We will monitor submissions for attempts to learn from or exploit the ranking
  process."
- The team wrote its own rule on 2026-09-21 (`docs/competitor_research_2026-09-21/results/14-validation-design.md`,
  A9): "any submission whose predictions are chosen to make the returned RMSE informative about the labels is a label
  measurement and should not be sent."
- On 2026-10-11 two planned leaderboard probes were dropped because of the organisers' sentence
  (`docs/final_2026-10-11/README.md`).

## 1. Every submission

Class:
- **M**: model or pipeline update, sent to improve the score.
- **A/B**: leaderboard comparison of one change against an earlier file, called "LB A/B" in the ledger.
- **D**: diagnostic or measurement (the ledger says probe, isolation, decomposition, test or check).
- **R**: post-processing rule on a small number of rows.

Several sessions shared the version counter; two numbers were used twice (§3).

| v | uploaded (UTC) | LB | class | ledger exp_id | what the ledger says |
|---|---|---|---|---|---|
| 1 | 09-10 17:33 | 345.3520 | M | v1_baseline | backfilled (uploaded before the ledger existed) |
| 2 | 09-10 19:25 | 312.0237 | M | v2_two_clock | backfilled |
| 3 | 09-10 22:21 | 306.2942 | M | v3_two_clock_weather | backfilled |
| 4 | 09-11 00:42 | 305.0270 | M | v4_linear_tree | backfilled |
| 5 | 09-14 09:54 | 325.2236 | D | main-f9d57a661f | "submitted to get a third CV<->LB pair and test whether local CV predicts the leaderboard at all" |
| 6 | 09-15 14:55 | 350.6249 | M | main-63086f0648 | weather bug fixed, model B 127 leaves, 3 seeds |
| 7 | 09-15 15:53 | 342.5933 | M | main-rebuilt-after-v6 | "v6 rebuilt without the nine rejected screening columns it shipped by accident" |
| 8 | 09-15 15:58 | 343.7964 | D | det_a-cfc9417c52 | "Submitted to find out whether the code itself regressed" |
| 9 | 09-15 22:00 | 306.4555 | D | probe-v8-matched-v4-unmatched | "Diagnostic probe … Splits the 32 s gap … into matched and unmatched halves, which offline analysis cannot do without labels." |
| 10 | 09-17 06:16 | 284.3584 | D, then M | submit_v10_reparam | "probe: v4's 339,551 matched predictions kept byte-identical, only the 5,290 unmatched rows replaced … The difference from v4's 305.027 is therefore the model-B change alone." The change was kept. |
| 11 | 09-19 01:54 | 293.9780 | R / D | v11-rollover-mixture-probe | "probe on v10: identical on 344,830 rows, 11 rows replaced by the mixture … floor -8.38 s if none of the 11 is a rollover; break-even at pi = 0.19" (§2.1) |
| 12 | 09-22 11:36 | 284.0450 | R | v12-lobt-bound-from-v10 | LOBT bound, a projection onto an identity window, on 34 rows. The ledger `kind` is "post-processing probe; no model was run". Pre-registered range met. |
| 13 | 09-23 06:15 | 282.7371 | D | v13-single-config-matched-probe | "Matched-lane probe … this is a measurement, not an improvement attempt, and both outcomes are informative" |
| 14 | 09-24 05:36 | 279.5234 | M | main_gap | `stand_arrival_gap` adopted |
| 15 | 09-28 04:27 | 279.5397 | M | main_v15 | dead-column removal |
| 16 | 09-28 18:47 | 279.2590 | M | main_flseq | fl / seq feature groups |
| 17 | 09-29 10:43 | 272.1788 | M | adsb-override-on-v14 | ADS-B pushback override. "The OOF-to-board carry-over is untested - this upload is the test." |
| 18 | 09-29 23:08 | 272.0002 | M | adsb-override-on-v16 | same override on v16 |
| 19 | 09-30 02:43 | 272.2934 | D | osm-flseq-adsb | "Board probe … OOF verdict REJECT (LIRF +9.65 s); submitted because the board scores the full final set" |
| 20 | 09-30 04:15 | 271.8800 | M | hybridH2-adsb | lane hybrid H2 + override; "lane choice post hoc" |
| 21 | 09-30 10:33 | 254.7657 | M | H2-adsb-learned | learned ADS-B residual replaces the rule; "Pre-registered ADOPT" |
| 22 | 09-30 12:08 | 254.2739 | M | plainy-adsb-learned | plain-target member |
| 23 | 09-30 22:05 | 252.4287 | M | plainy-cb-adsb-learned | CatBoost member |
| 24 | — | — | — | mbs_mix255 | failed before upload; nothing uploaded |
| 25 | 10-01 00:08 | 272.0903 | A/B | mbs_mix255 | "LB A/B probe of the model B capacity blend: only the 5,180 unmatched rows differ from v20" |
| 26 | 10-01 04:44 | 278.8726 | A/B | flwin_both | "LB A/B of the fl_nm_rate_other window fix against v16" |
| 27 | 10-01 08:08 | 278.7431 | A/B | stack(main_flseq, clip1800) | "LB A/B of the clipped-member segment stacker against v16" |
| 28 | 10-02 00:00 | 252.4111 | M | blend-l2-adsb-learned | forward NM windows (L2) in the matched members |
| 29 | 10-02 00:15 | 252.5307 | M | v28+l5delta | v28 + L5 member delta |
| 30 | 10-02 00:25 | 252.4077 | M | v28 + window fix on ADS-B-untouched unmatched rows | graft of v26's model B onto 2,874 unmatched rows; "evidence: v26 vs v16 LB -0.39 with the same change" |
| 31 | 10-02 00:26 | 279.2069 | D | v16 + v26 matched rows (decomposition probe) | "Decomposition probe … split v26's -0.39 into matched vs ADS-B-corrected unmatched rows" |
| 32 | 10-02 03:04 | 251.7333 | M | blend-l2-pa-adsb-learned | per-airport member |
| 33 | 10-03 01:34 | 250.9496 | M | blend-l2-pa-opt-adsb-learned | retuned members |
| 34 | 10-05 02:19 | 246.7493 | R | v33-posjoin-overlay | positional ADS-B overlay on 430 unmatched rows, "Mostly EHAM 2026-01-02..07 snow days" (§2.2) |
| 35 | 10-05 04:55 | 245.9350 | M | mlp-adsb-posjoin | MLP member + v34 overlay |
| 36 | 10-05 06:43 | 244.0718 | M | v35-overlays-c1 | posjoin residual (885 rows), LIRF copy blend, LOBT clip (18 rows) |
| 37 | 10-05 06:45 | 242.5901 | M | v35-overlays-months-c2 | 4-month residual |
| 38 | 10-05 10:34 | 242.0919 | M | v37-v2params-c3 | residual V2 settings |
| 39 | 10-06 00:46 | 239.5208 | M | v38-appear-fill-v2 | appear tier |
| 40 | 10-06 01:16 | 238.7712 | M | v39-7m-pos-12m | 7- and 12-month residual swaps; LFPG choice "post hoc" |
| 41 | 10-06 02:16 | 239.5174 | R | v40-lirf-label-classes | LIRF unmatched label classes on 29 rows (§2.3) |
| 42 | 10-06 15:00 | (overwritten) | M | v40-plus-r6cbopt-delta | R6 reweight + retuned CatBoost delta. Overwritten by the next line. |
| 42 | 10-07 02:06 | 236.4369 | M | v40-batch-c12 | C12 batch |
| 43 | 10-07 03:29 | 238.6626 | D | iso-apron | "isolation … isolate which part of v42 (238.7644) helped or hurt" |
| 44 | 10-07 03:29 | 237.4070 | D | iso-c10 | isolation, same purpose |
| 45 | 10-07 03:29 | 238.3508 | D | iso-pb | isolation, same purpose |
| 46 | 10-07 04:20 | 236.4631 | M | c12-appear12 | C12 + 12-month appear delta |
| 47 | 10-08 01:21 | (overwritten) | M | c12-appear12-old | C12 + 12-month appear, rebuilt. Overwritten by the next line. |
| 47 | 10-08 01:28 | 234.4831 | M | v42-flfixm-seeds-ucmix | C15 |
| 48 | 10-11 14:47 | 234.4099 | M | c15-appear12-cov26eddf | C15 + 12-month appear + EDDF coverage delta (team best) |
| 49 | 10-11 14:56 | 248.6458 | D | finalbase-plus-adsb12-janjul | "Final-phase fallback check": the Jan/Jul rows of the 10-09 bucket `final.parquet` + ADS-B |
| final | 10-13 01:20 | — | — | — | dry run only (`prc.submit --final --dry-run`); nothing uploaded |

Counts: 50 uploads under 48 version numbers (v24 was never uploaded; v42 and v47 were uploaded twice each). A
submission can carry two classes.
- M: 33 uploads under 31 version numbers. This includes v10, which was sent as a probe.
- A/B: 3 (v25, v26, v27).
- D: 12 (v5, v8, v9, v10, v11, v13, v19, v31, v43, v44, v45, v49). v10 is also M and v11 is also R.
- R: 4 (v11, v12, v34, v41).

## 2. Rules derived by inspecting ranking-set rows

No ranking-set label is available to the team. "Inspecting" here means looking at the 2026 rows' inputs: times,
flight numbers, ADS-B tracks. It also includes reading leaderboard scores back into decisions about specific rows.

### 2.1 LIRF day-rollover rows (v11)

- `docs/investigation_2026-09-16/results/block-day-rollover.md` and
  `docs/data_audit_2026-09-18/results/forensics-rollover.md` tabulate individual 2026 LIRF unmatched flights by
  flight number, for example `TAY9LA` and `NOSOS431`.
- v11 replaced 11 such ranking rows with a mixture. The rule was LIRF, unmatched, `MVT − SCHED ≥ 43,200 s`, with a
  precision measured on 29 training rows.
- The score went from 284.3584 (v10) to 293.9780 (v11). The rule was not kept: v12 was built from v10.

### 2.2 Positional ADS-B overlay (v34; still in the chain)

- `docs/posjoin_2026-10-05/README.md` motivates the overlay with the EHAM 2026-01-02..07 snow days in the ranking set:
  "v33 predicts about 2,400–2,500 s for those flights. The ADS-B ground tracks give a median of 3,350–3,750 s."
- The row list (430 Jan/Jul ids) is committed: `docs/posjoin_2026-10-05/replacements_v33.csv` (745cef9).
- The rule is generic (unmatched, no callsign join, a push signature, no later stop ≥ 900 s; long stops at
  w = 0.75). Its weights were set on 2025 rows.
- After v34 the notes say: "the board gain of v34 (2,090 MSE) exceeded the estimate, which says the rule worked
  there" (`docs/posjoin_2026-10-05/deeper_checks.md`).

### 2.3 LIRF label classes that name 2026 flights, and the v41 back-solve

- `analysis/local_lirf_artefacts_apply.py` (48db3bb), rule **C7r** ("one-day" class): per-row q values were revised
  for named 2026 flights. The docstring says: "TAY9LA-type rows (no midnight crossing) q = 0.05; rows whose re-dated
  block is > SCHED + 4 h (NOSOS431, +6.7 h) q = 0.15".
- `docs/lirf_wrap_2026-10-06/README.md`: "Per 2026 row: TAY9LA does not cross midnight, so q 0.29 → 0.05. NOSOS431
  lands at SCHED + 6.7 h, so q 0.65 → 0.15." Also: "v40 predicts all six 2026 rows of this kind as schedule copies".
- `analysis/local_lirf_wrap2.py` line 35 hard-codes five 2026 flights and their q:
  `{"WMTT3508": 0.65, "ITYTY672": 0.65, "SVII3006": 0.65, "TAY9LA": 0.05, "NOSOS431": 0.15}`.
- v41 = v40 + these classes on 29 rows. It scored 239.5174 against v40's 238.7712.
- **Back-solve**, from the same README: "Back-solving the score: about 1 of the 3 high-q one-day rows was a wrap (the
  2025 rates expected 2; P(≤ 1 of 3) ≈ 0.35). The one-day blend is dropped." That is, the board score was used to
  infer the labels of three specific ranking rows. The class was then dropped (0d4b1db).
- The final chain does not apply C7r. `docs/final_2026-10-11/apply_overlays.py` says: "rule C7r, the 'one-day'
  class, is NOT applied: dropped after v41".
- It keeps the two other LIRF classes. Their rates come from 2025:
  - the time-zone class (per-prefix q; 6 rows in Jan/Jul 2026);
  - the copy band (7–13.5 h late; 17 rows).

### 2.4 Leaderboard scores read back into decisions

| what | source |
|---|---|
| v9 split the board gap into lane halves, "which offline analysis cannot do without labels" | ledger v9 |
| v10 and v13 isolated one lane each by keeping the other byte-identical | ledger v10, v13 |
| v31 split v26's board change into matched and unmatched rows | ledger v31, `docs/v28_graft_2026-10-02/plan.md` |
| v30 grafted v26's model B onto 2,874 rows because of the v26 vs v16 board difference | ledger v30 |
| v43–v45 isolated the parts of v42. They were sent after a misread board value; the notes add: "the isolation submissions were not needed. They still give the attribution" | `docs/batch_2026-10-06/README.md`, b6c627e |
| board-transfer ratios (consistency fixes about 2× their CV estimate, ADS-B add-ons 10–35 %) were recorded and used in planning | `docs/SOLUTION_OVERVIEW.md` §6 |
| v49 scored the Jan/Jul rows of the 10-09 bucket `final.parquet` plus ADS-B; the result decided "that base is not usable" | ledger v49, 169e7f4 |

### 2.5 Other row-level steps that touch few ranking rows

These were derived on 2025 data and applied to the ranking rows they select:
- LOBT clip: 17–34 rows per build;
- LIRF time-zone class: 6 rows;
- LIRF copy band: 17 rows;
- ucmix: 356 LIRF unmatched rows.

Per-row delta files keyed to ranking ids are committed under `handoff/`.

### 2.6 Final-phase steps after the leaderboard closed

- **+1×δ re-anchor** (`docs/final_2026-10-11/add_delta_janjul.py`; recorded as the user's decision in a94067b).
  - It adds to the Jan/Jul matched rows the difference between `MVT_TIME_UTC_mvt` in `final_ranking.parquet` and in
    `ranking.parquet` (±12 s per row).
  - It comes from comparing the two releases of the ranking-set rows, not from labels.
  - It is not in the runbook's file. It is in `handoff/2026-10-13/optional_janjul_reanchor_plus1delta.parquet` and in
    the `*_plus1delta` files of `handoff/2026-10-14/`.
- **LIRF copy-overlay gate** (`handoff/2026-10-14/README.md`, `docs/final_phase_gate/`).
  - It removes the copy overlay on 2,486 LIRF matched rows where `|LOBT − SCHED| > 3606 s`.
  - It was validated on 2025. The README notes that "about 10 rows" carry 87 % of the change.
  - It is not in the runbook's file.

## 3. Process facts

- **Version collisions.** In both cases the later upload replaced the scored object, and the board shows the later
  file's score (`docs/batch_2026-10-06/README.md`, cf0f289, 33327e5).
  - v42: model-b's upload at 10-06 15:00 (660ca56) was replaced by laughing-cray's C12 at 10-07 02:06.
  - v47: beautiful-davinci's upload at 10-08 01:21 (scored 236.3563 per `handoff/2026-10-08/ON_C15.md`) was replaced
    by laughing-cray's C15 at 01:28.
- **Unrecorded upload.** A `resilient-hedgehog_final.parquet` dated 2026-10-09 16:02Z was in the team bucket.
  - No ledger line or commit records who made it (`docs/final_phase_2026-10-11/RUNBOOK_final_submit.md`).
  - The organisers' scorer marked it Failed with an error on their side.
- **Choices labelled "post hoc" in the notes:**
  - lane hybrid H2 (d912c37, shipped as v20);
  - OSM distance nulled at LIRF (221779d);
  - LFPG kept on the 7-month residual (cb0649d, in v40);
  - 12-month appear candidate B (1f20061, not submitted).
- **After v41** the team rule became: "Few-row, heavy-tailed gains are rejected" (`docs/SOLUTION_OVERVIEW.md` §6).
