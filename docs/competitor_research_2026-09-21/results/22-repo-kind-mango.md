# Slot 22 — repo `kind-mango`: signed-delta ablation ledger

**Source repo** [S]: `https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango`
Cloned to `/tmp/prc-kind-mango` (HEAD at clone time, 2026-09-21). Team `kind-mango`,
self-reported best **287.33 s** (v57), worse than resilient-hedgehog's 284.3584.

Primary files read in full:
- `/tmp/prc-kind-mango/RECAP.md` (414 lines) — the ledger
- `/tmp/prc-kind-mango/README.md` (430 lines) — model card, ethics, ruled-out list
- `/tmp/prc-kind-mango/docs/MODEL_ANALYSIS.md` (1262 lines) — fifteenth-pass audit, findings + measures
- `/tmp/prc-kind-mango/src_v3/train_v54_base.py`, `predict_v54.py` — the anchored-offset failure, as code
- `/tmp/prc-kind-mango/src_v3/postprocess.py`, `train_v51_base.py`, `train_v60_base.py`
- `/tmp/prc-kind-mango/src/check_coverage.py`, `src/train_lgbm_v21.py`, `src_v3/arrival_mirror.py`
- `/tmp/prc-kind-mango/models/*.json`, `/tmp/prc-kind-mango/submission/*.check.json`

**Evidence policy.** Everything marked [S] is a direct quote or arithmetic on a file in
their repo. Their score is worse than ours: their *measurements* are evidence, their
*conclusions* are claims. I mark claims as claims. [X] = my inference, never dressed up.

---

## 0. Corrections to the brief I was given

Three numbers in my tasking do not survive contact with the repo. Flagging first so the
rest is read correctly.

1. **"v10 560.91 -> v21 330.12 (-230.79) from fixing three bugs"** — [S] overstated.
   `RECAP.md:60`: *"Fixing those (Step 2, v21) took live from **560.91 to 430.45** — a
   **130 s drop**."* The three-bug fix is worth **-130.5 s**, not -230.79. The further
   430.45 -> 330.12 comes from four *additional*, separately-ledgered changes (Step A band
   lookup -58, v20 turnaround/disruption/ITY340 -24.9, v21 linear_tree+leak fixes -15.6,
   v18 6.3 rule -1.77). Note also that `RECAP.md` labels the same live score 430.45 as
   both "v13" (scoreboard row) and "v21 = Step 2" (Breakthrough section) — their own tag
   bookkeeping is inconsistent across that boundary [S, `RECAP.md:60,116,153`].
2. **"v26 -26.41 by dropping `ec_*`/`opdi_*`"** — [S] the ledger says **-13.14 s**
   (316.85 -> 303.71), `RECAP.md:103,223`. There is no -26.41 anywhere in the repo. Also
   the counts: **38 `ec_*` + 18 `opdi_*`** (`RECAP.md:103`, `README.md:105`), not 38+9;
   a *separate* later experiment (v49) used 38 `ec_*` + 9 `opdi_*` + 9 `opdi_live_*` = 56.
3. **"LSZH January-2026 traffic 25 % below 2025"** — [S] this is **not traffic**. It is
   the count of **OPDI `taxi_out_v2` records** (an external open dataset) at LSZH:
   5,630 in Jan-2026 vs 7,542 in Jan-2025 = -25.4 %
   (`models/coverage_monitor.json:opdi_taxi_out_counts.LSZH`; `src/check_coverage.py:1-13`
   defines it as *"counts the taxi_out_v2 records per airport and month"*). It is
   **external-feed coverage drift**, not a demand/traffic drop. Their own July figure
   (12,588 vs 11,770, +7 %) makes a traffic reading implausible.

The `+35.86` anchored-offset number and the `-8.32`/`-5.99` v51 number are the items to
check next; the v51 figure is also different from my brief (see §4).

---

## 1. The complete ledger (transcribed, not summarised)

### 1.1 Live scoreboard, `RECAP.md:62-125`

Verbatim tags, scores and the accept/reject verdict. "Live" = leaderboard RMSE in seconds.
Deltas are as the repo states them (each is against the stated reference, not against the
row above).

| tag | live RMSE | signed delta | verdict | what changed |
|---|---|---|---|---|
| v60 | 289.58 | **+2.25 vs v57** | REJECTED | MD4 leave-one-month-out operator encoders (OOF at training, full-year map at serving) |
| v59 | 287.70 | **+0.37 vs v57** | REJECTED | MF4 arrival-taxi-in drift features (1d/7d rolling median minus 2025 fit-month per-airport median), 4 cols, base retrained at 121 features |
| v58 | 288.62 | **+1.29 vs v57** | REJECTED | MB8 applied to operator target encoders: refit `features_operator.fit_encoders` on all 12 months (was months 2..12 excl. 7) |
| **v57** | **287.33** | **-0.17 vs v56** | **ACCEPTED — team best** | MB8 on `plan_taxi_res` route medians: recompute on all 12 months of 2025 clean rows; 5,883 routes (v45 had 5,619) |
| v56 | 287.50 | **-0.33 vs v55** | ACCEPTED | MB8 on the LIRF `p_fb` gate: refit `lgbm_p_fb_lirf` + isotonic on all 12 months of LIRF rows. Full-year AUC 0.9256, log-loss 0.2779 (in-sample). 25,044 LIRF rows moved, mean -16.81 s |
| v55 | 287.83 | **-0.42 vs v51** | ACCEPTED | MB8 on `R_norm_LIRF`: refit the 5 boosters on all 12 months of LIRF genuine rows; iters scaled by 1/0.88. 26,795 rows moved, mean -0.27 s. "Ceiling was ~0.4 s (LIRF weight 26k/344k) and it landed at that ceiling" |
| v54 | 324.11 | **+35.86 vs v51** | REJECTED | **MB1 anchored-offset target `y - anchor`**, anchor = `mvt_eobt1` else `sched_delay`, on top of MB3. See §3 |
| v52 | 288.36 | **+0.11 vs v51** | REJECTED (noise) | MB3+MB4: cyclic doy / local hour / public holiday added, numeric month dropped |
| **v51** | **288.25** | **-11.06 vs v41, -5.99 vs v48** | **ACCEPTED** | **MB3: base retrained on served rows only.** Excluded 160,706 LIRF rows + 2 non-LIRF `y>80,000` rows; retuned Optuna, 3 seeds, 12 months, iter scaling 1.259 (767/2265/1278) |
| v50 | pending | — | never scored | MS3 member-disagreement median on top of v48; moved 15 rows net |
| v49 | 294.65 | **+0.42 vs v48** | REJECTED | MD2+MH1: LIRF head rebuilt without the 56 drifted columns (38 `ec_*` + 9 `opdi_*` + 9 `opdi_live_*`). Paired LIRF-only gain -2,405 MSE did **not** transfer |
| v48 | 294.24 | **-2.33 vs v47** | ACCEPTED | MS1 per-airport upper bounds + MS2 clean floor, post-hoc on the v47 parquet. MS1 moved **5** rows, MS2 moved 104. Zero training cost |
| v47 | 296.57 | **-2.74 vs v41** (-2.57 vs v46) | ACCEPTED | L2: 12-month refit of the v46 stack, no early-stop split, iters × 1/0.88 (609/1799/1015 -> 693/2045/1154) |
| v46 | 299.14 | **-0.17 vs v41** | ACCEPTED (marginal) | L1 Optuna retune, 30 trials, `linear_tree` pinned; leaves 436, min_data 291, lr 0.0183, ff 0.5629, linear_lambda 0.0056. Paired -1,864 MSE; only -102 MSE landed |
| v45 | pending | — | never scored | L9 `plan_taxi_res` clipped ±3600 s; 5,619 route medians; paired -650 MSE clean |
| v44 | pending | — | never scored | L3.a p25/p75 of neighbour `mvt_eobt1` at apt30/apt60/rwy30; paired -552 MSE clean |
| v41 | 299.31 | **-0.65 vs v40** | ACCEPTED | `R_norm_LIRF` retrained with the 13 tempo columns (paired -228) + 4,431 s cap removed (paired -187) |
| v40 | 299.97 | **-1.90 vs v33** | ACCEPTED | 13 tempo / order / stand-gap / queue columns on the base. First ship under 300 |
| v39 | 352.19 | **+50 vs v33** | REJECTED | Cold-start rewrite (`src_v2/`). 62-73 % of the +32,911 MSE on **one** LIRF row with `sd = 94,560` (v33 served 88,718; v39 served 3,262 after the audit removed the ITY340 rule) |
| v38 | not uploaded | gate failed | REJECTED | `ARVT_1_flt` planned-time features; seed 43 collapsed at iteration 6 in two identical runs; 3-member CLEAN 277.27 vs v26 266.46 |
| v37 | 302.52 | **+0.65 vs v33** | REJECTED | 5-seed `p_fb_LIRF` mean. Priced -151 MSE; live miss of 547 MSE |
| v36 | 302.05 | **+0.18 vs v33** | REJECTED | F7 zero-clip repair (drop per-member clip at 0, then fill exact-zero rows with per-airport positive median). 110 rows differ, 44 zeros filled |
| v35 | 302.09 | **+0.22 vs v33** | REJECTED | v34's honest `p_fb` + v33's all-month scoring artefacts. Isolated the regression to the honest classifier, not the lookups |
| v34 | 302.07 | **+0.21 vs v33** | REJECTED (hygiene ship) | Hold-out leak removed from LIRF stats; refit on months {2..12}\{7}. `NORMAL_MEAN_LIRF` 1150 -> 1094.41; band table 127 -> 82 cells |
| **v33** | **301.87** | **-0.11 vs v30** | ACCEPTED | 5-member `R_norm_LIRF` mean (seeds 42-46), 3-seed base unchanged. **Predicted 301.86, landed 301.87** |
| v32 | 302.11 | **+0.13 vs v30** | REJECTED | v33 + 7-member base mean. A_5 = 74 MSE matched the audit; A_7 = 5,417 MSE was 8× its hold-out estimate |
| v31 | 302.41 | **+0.43 vs v30** | REJECTED | Cleaned invalid ARR taxi-in labels before congestion aggregation |
| v30 | 301.98 | **-1.28 vs v29** | ACCEPTED | LIRF-only encoders for LIRF models (fix: all-airport encoders were being served to LIRF-trained models) + mutually-exclusive band table |
| v29 | 303.26 | **-0.45 vs v26** | ACCEPTED | 3-seed base + `R_norm_LIRF` clip 4,431 s + ITY340 constant formula; Step A base-term clip removed (it had regressed +124 MSE) |
| v28 | 304.48 | **+0.77 vs v26** | REJECTED | 7-seed `R_all_v27` alone (isolates the ensemble) |
| v27 | 304.03 | **+0.32 vs v26** | REJECTED | 7-seed + drop 3 dead features + Step 2 fixes |
| **v26** | **303.71** | **-13.14 vs v24** | **ACCEPTED** | **Drop 38 `ec_*` + 18 `opdi_*`**, retrain 3 seeds. See §4.1 |
| v25 | 375.33 | **+58.5 vs v24** | REJECTED | Impute `ec_*` with 2025 medians. "proves `ec_*` are actively harmful, not just missing" |
| v24 | 316.85 | **-0.65 vs v23** | ACCEPTED | 3-seed mean with honest 12 % random early-stop split |
| v23 | 317.50 | **-2.8 vs v22** | ACCEPTED | Fallback-rate encodings on `p_fb` (7 keys × 2 = 14 features, OOF leave-one-month-out, K=30 smoothing). `fbrate_flt_prefix` rank 1/167 |
| v22 | 320.31 | **-9.8 vs v21** | ACCEPTED | LIRF regime head (`R_norm_LIRF` + calibrated `p_fb` mixture) + refined 11-bin band table; retired the 6.3 rule |
| v21 | 330.12 | **-15.6 vs v20** | ACCEPTED | `linear_tree` + signed-log copies + OPDI-live 600 s lag leak fix + turnaround `BLOCK_TIME` fix + temperature |
| v20 | 345.70 | **-24.9 vs v18** | ACCEPTED | Turnaround + disruption meters + ITY340 rule |
| v19 | 359.46 | **-11.2 vs v18** | ACCEPTED | Per-airport combined: v18 at LIRF/EHAM, R_norm+detector at EGLL/LEBL/LTFM, R_norm alone elsewhere |
| v18 | 370.63 | **-1.77 vs v16** | ACCEPTED | Section 6.3 rule for LIRF null-flight `sd` 3,600-14,400 (shrink 0.6) |
| v17 | 377.47 | **+5 vs v16** | REJECTED | Step C LIRF+EGLL fallback-detector soft mix |
| v16 | 372.40 | **-58 vs v13** | ACCEPTED | Step A LIRF null-flight band rule |
| v13 | 430.45 | **-130.5 vs v10** | ACCEPTED | **The three-bug fix** (unfiltered labels + OBT deltas). Tagged "v21 = Step 2" in the Breakthrough text |
| v14 | 431.96 | +1.5 | REJECTED | 4-model ensemble refit on 12 months |
| v15 | 625.34 | +195 | REJECTED | Step 3 tail mixture. "Mixture blew up on live" |
| v10 | 560.91 | — | earlier best | 5-model NNLS on the old broken metric |
| v9 / v7 / v12 / v6 | 561.04 / 561.06 / 561.15 / 561.19 | — | — | pre-fix plateau |
| v3-v5, v8 | 561.5-562.4 | — | — | pre-fix plateau |
| v1-v2 | 562.3-562.4 | — | — | pre-fix plateau |

### 1.2 Levers gated offline and never uploaded, `RECAP.md:82-89, 380-410`

Paired hold-out price, negative MSE = improvement. Their gate thresholds in parentheses.

| lever | paired served-clean price | gate | verdict + stated reason |
|---|---|---|---|
| L7 weather at EOBT_1 (`tmpc/vis/wind/wx_precip/deicing_gate` joined at pushback) | **-293 MSE** (total -367 across 4 classes) | 300 | REJECTED, short by **7 MSE**. All four served classes improved. Kept open |
| L8 calendar (public holiday / weekend flags, `holidays` 0.104) | **-144 MSE** (total -105) | 200 | REJECTED, short by 56. Holiday coverage 2.6 %, weekend 28 %; hour/dow categoricals already capture it |
| L3.e arrival taxi-in residual (prev-30-min median minus 2025 fit-month median) | **+102 MSE** (genuine regression) | 300 | REJECTED. Airport medians sensible (LTFM 772 s, LIRF 592, LEBL 209) but no residual signal beyond congestion/tempo/`plan_taxi_res` |
| L3.c neighbour EOBT by (ADEP, stand_prefix) 30 min | **+77 MSE** | 300 | REJECTED; stand categorical + rwy30 tempo already capture terminal geometry |
| L3.b p25/p75 of neighbour `mvt_iobt` | **+16 MSE** (total +77) | 300 | REJECTED; IOBT redundant with EOBT_1 once EOBT quartiles are in |
| L3.d neighbour EOBT by (airport, operator) 120 min | **+54 MSE** (total +128) | 300 | REJECTED; captured by the 16 operator-historic encoders |
| L4 constant-leaf booster meaned 0.7/0.3 with v44 | **+64 MSE** (total +196) | 400 | REJECTED. v44 alone CLEAN 256.22; constant-leaf alone 262.86; mean 256.64 — "the weaker class drags the mean" |
| L5 `R_norm_LIRF` with `FB_TOL = 5` on the fit mask | LIRF clean **+3,546**, fallback -1,200, total -2,421 | 200 | REJECTED. The 5-s mask pulls near-fallback rows into training and hurts the clean predictor |
| v42 forward take-off order on the base | -93 MSE | — | set aside (also a policy risk, L4 finding) |
| L3.f runway-use shares | — | — | skipped; duplicates `runway_diversity_prev_30m` |

### 1.3 "What does not work", `RECAP.md:175-199` — hold-out vs live, verbatim

| approach | hold-out delta | live delta | their stated reason |
|---|---|---|---|
| DQ classifier | AUC 0.51 | — | Random |
| SSL drift correction | ~0 | + | Hurt live |
| Quantile regression | — | — | No signal |
| LIRF specialist | ~0 | — | Same convergence |
| Tail mixture (old formulation) | — | — | Wrong economics for RMSE |
| Extended OPDI event counts (v18) | ~0 | — | Redundant |
| VRS / openap metadata | ~0 | — | Duplicates categorical |
| Historical climatology 2022-24 (v19) | ~0 | — | Marginal |
| Aggressive cleanup (v20) | ~0 | +0.24 | Only 60 rows removed |
| v20-swap ensemble (v12) | -0.03 | +0.24 | v20 too close to v15 |
| Fixed arrival context (Step 4, v23) | **-1.5** | **~0** | Right in principle, no live gain |
| OPDI ratio-to-median (Step 5, v24) | **-1.5** | **~0** | Right in principle, no live gain |
| 4-model seed/ff ensemble (Step 6, v25) | -0.1 | **+1.5** | 12-month refit hurts live |
| Runway configuration categorical (Step 7) | **+7.6** | — | High cardinality overfits |
| **Fallback-and-24h classifier (Step 3)** | **-40** | **+195** | **Classifier does not generalize to 2026** |
| Step A LIRF null-flt band rule (v16) | -83.6 | **-58** | Deterministic lookup, big win |
| Step A variant, gate 3600 extended | +9.8 | — | Normal rows in band get wrongly boosted |
| **Step C LIRF+EGLL detector soft mix (v17)** | **-8** | **+5** | **Classifier overfits 2025 hold-out** |
| Section 6.3 LIRF no-flight-record group | -2.3 | **-1.8** | Shrink 0.6 on detector; mix vs group-mean |
| Section 6.2 `R_norm` alone | LIRF clean -247 | in v19 | Removes v21's bimodal bias at LIRF |
| v20 combined per-airport (v19) | -2.25 | **-11.2** | Per-airport winner selection |

### 1.4 Paired-price vs live-outcome calibration table [S, `MODEL_ANALYSIS.md:119-137`]

This is, for our purposes, the single most useful table in the repo. MSE, negative = better.

| ship | change | paired price | **live delta** |
|---|---|---|---|
| v32 | 7-seed base, 5-member `R_norm` | -1,278 | **+78** |
| v33 | 5-member `R_norm` only | -74 | **-66** |
| v37 | 5-seed gate mean | -151 | **+396** |
| v40 | 13 tempo columns on the base | -805 | **-1,146** |
| v41 | tempo on `R_norm`, no cap | -415 | **-390** |
| v44+v45+v46 | quartiles + plan residual + retune (stacked) | **-1,864** | **-106** |
| v47 | 12-month refit of the base | not priced | **-1,530** |

Their conclusion [claim]: *"Single feature changes against a same-run control landed.
Stacked changes did not. The largest recent gain had no offline price."* Their stop rule:
*"Do not ship a change priced under the 268 MSE lottery"* (`RECAP.md:409`).

---

## 2. The 97-feature catalogue

The shipped v26/v40 base reads **97 features**; v47 reads 117 (the 97 plus 20 extras
joined by movement id). [S] `models/lgbm_r_all_v47.meta.json` carries the ordered 117-name
list; `README.md:146` states *"97 features. v26 dropped the 38 `ec_*` and 18 `opdi_*`
columns."* Subtracting the 20 extras (last 20 entries of the v47 list) from 117 gives
exactly the 97 [X — arithmetic on their two files, they never print the 97-name list
itself].

Group counts are from the v21 model card, `README.md:160-193`, which enumerates the full
153-column pre-v26 set; 153 - 38 `ec_*` - 18 `opdi_*` = 97.

| # | group | count | features (exact names, `meta.json` order) | definition |
|---|---|---|---|---|
| 1 | categoricals | 10 | `ADEP_mvt`, `ADES_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `AIRCRAFT_TYPE_mvt`, `WK_TBL_CAT_flt`, `MARKET_SEGMENT_flt`, `AIRCRAFT_OPERATOR_flt`, `FLIGHT_RULE_mvt`, `FLIGHT_TYPE_flt` | airport, destination, runway, stand, aircraft type, wake category, market segment, operator, flight rule, flight type. Raw strings, **shared across airports** (their finding D6) |
| 2 | time & schedule | 14 | `hour`, `dow`, `month`, `sched_delay`, `sd_mod_86400`, `mvt_eobt1`, `mvt_iobt`, `eobt1_sched`, `eobt1_iobt`, `sched_delay_slog`, `mvt_eobt1_slog`, `mvt_iobt_slog`, `eobt1_sched_slog`, `eobt1_iobt_slog`, plus `flt_null`, `flt_id_null` | `src/train_lgbm_v21.py:66-96`: `sched_delay = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` (**unclipped** since v21); `sd_mod_86400 = sched_delay mod 86400`; **`mvt_eobt1 = MVT_TIME_UTC_mvt - EOBT_1_flt`**; `mvt_iobt = MVT_TIME - IOBT_flt`; `eobt1_sched = EOBT_1 - SCHED`; `eobt1_iobt = EOBT_1 - IOBT`. `_slog` = signed-log copies (added v21 with `linear_tree`). `flt_null = AIRCRAFT_OPERATOR_flt isna`; `flt_id_null = FLIGHT_ID_mvt isna` |
| 3 | METAR | 18 | `sknt`, `gust_kt`, `wind_cross_kt`, `wind_head_kt`, `vis_km`, `low_vis`, `very_low_vis`, `ceiling_ft`, `low_ceiling`, `wx_precip`, `wx_snow`, `wx_thunder`, `wx_freezing`, `tmpc`, `dwpc`, `dewpt_spread`, `ice_accretion_1hr`, `deicing_gate` | Iowa State ASOS, joined at `MVT_TIME`. Crosswind/headwind resolved on the **active runway**. `deicing_gate = wx_precip * (tmpc < 3)`. `ice_accretion_1hr` is **all NaN — dead** (`RECAP.md:290`) |
| 4 | congestion v2 | 13 | `dep_load_prev_15m/30m/60m`, `arr_load_prev_15m/30m/60m`, `dep_same_rwy_prev_15m/30m/60m`, `dep_queue_next_10m`, `arr_same_rwy_prev_15m`, `arr_same_rwy_next_10m`, `arr_taxi_in_mean_60m` | `src/features_congestion_v2.py`. Arrival counts keyed on `ADES_mvt`; same-runway ARR/DEP counts; rolling arrival taxi-in mean (missing taxi-in filled **0**, their D7); next-10-min queue. Two columns (`dep_queue_next_10m`, `arr_same_rwy_next_10m`) count movements **after** the scored take-off — their own L4 eligibility-risk finding |
| 5 | operator historic encoders | 16 | `openc_taxi_median`, and median/count/std for each of `openc_op_apt_*`, `openc_op_apt_rwy_*`, `openc_op_apt_hbin_*`, `openc_op_*`, `openc_op_apt_stand_*` | `src/features_operator.py:31-46`. Median, count, std of historic taxi time per key, `MIN_COUNT = 20`. **Fitted and applied in-sample** — their L1 leak finding |
| 6 | physical / geometry | 7 | `taxi_dist_m`, `taxi_dist_known`, `stand_lat`, `stand_lon`, `rwy_thr_lat`, `rwy_thr_lon`, plus `taxi_path_length_m`, `n_turns`, `path_vs_haversine` | OSM `aeroway=parking_position`/`taxiway`/`runway`. Haversine stand->runway-threshold, plus the real taxiway-graph shortest path, turn count, and their ratio |
| 7 | advanced physical | 5 | `sched_pushback_load_15m`, `runway_diversity_prev_30m`, `secs_since_last_dep_same_rwy`, `rwy_bank_intensity_5m`, `ades_arr_atfm_delay_today` | `src/features_advanced.py`. Gap since last DEP defaults to **3,600 s** when absent. `ades_arr_atfm_delay_today` has only **4-10 % row coverage** and is a same-day destination total that accrues after take-off (their L5) |
| 8 | turnaround | 6 | `ground_time`, `slack_eobt`, `slack_sched`, `in_delay`, `taxi_in_prev`, `link_ambiguous` | `src/features_turnaround.py`. Links each departure to the last arrival on the same stand, joining on `BLOCK_TIME_UTC_mvt` (v21 fix: was landing + clipped taxi-in). Reach-back 12 h |
| 9 | disruption meters | 4 | `dep_sd_mean_60m`, `dep_fnull_60m`, `arr_txi_mean_60m`, `arr_txi_p90_60m` | 60-min window before take-off: mean schedule delay of departures, null-record share, arrival taxi-in mean and p90 (kept to 30-7,200 s, unlike group 4) |
| — | **subtotal = the 97** | **97** | | |
| + | tempo / order / stand-gap / queue (v40) | 13 | `nb_eobt_med/mean/cnt_apt30`, `..._apt60`, `..._rwy30`, `order_later_eobt_30`, `order_total_30`, `stand_gap`, `queue_eobt_mvt` | neighbour `mvt_eobt1` statistics over airport-30min / airport-60min / same-runway-30min windows |
| + | EOBT quartiles (v44, L3.a) | 6 | `nb_eobt_p25/p75_apt30`, `..._apt60`, `..._rwy30` | p25/p75 of the same neighbour pool |
| + | plan residual (v45, L9) | 1 | `plan_taxi_res` | route median taxi from 2025 clean rows, residual, clipped ±3,600 s |
| | **v47 total** | **117** | | |

**Dropped in v26 and never restored** [S]: 38 `ec_*` (Eurocontrol daily ATFM per airport —
airport traffic ×3, all-pre-departure-delay ×2, ATC-pre-departure-delay ×2, each with a
lag-1 copy, plus others) and 18 `opdi_*` (9 raw OPDI event counts — entry-runway,
entry-taxiway, exit-parking at prev 15/30/60 min, leak-fixed by a 600 s lag; and 9
OPDI-live rolling taxi stats — mean/median/count of ACTUAL observed taxi times of other
completed flights over 30/60/120 min).

**Feature ranks in v21** [S, `RECAP.md:257-265`]: `mvt_eobt1` **rank 1 of 130** (highest
gain of all features), `sched_delay` rank 3, `mvt_iobt` rank 5, `sd_mod_86400` rank 6,
`eobt1_iobt` rank 10, `eobt1_sched` rank 12. Also `mvt_eobt1_slog` rank 8/153 and
`sched_delay_slog` rank 10 in v21; `fbrate_flt_prefix` rank 1/167 in the LIRF gate.

The LIRF head is a **different** feature set: 166 columns for `R_norm_LIRF`, 167 for the
`p_fb` gate, including all 38 `ec_*` (never removed there — their D1 finding).

---

## 3. The anchored-offset failure (v54, +35.86 s) — full mechanism

This is the item that matters most for us, so I read the code, not only the ledger.

### 3.1 What exactly the anchor was

[S] `src_v3/train_v54_base.py:44-47`:

```python
def _compute_anchor(dep: pd.DataFrame) -> np.ndarray:
    sd = dep["sched_delay"].astype(float).values
    mvt_eobt1 = dep["mvt_eobt1"].astype(float).values
    return np.where(np.isnan(mvt_eobt1), sd, mvt_eobt1)
```

with, from `src/train_lgbm_v21.py:72,93-95`:

- `mvt_eobt1 = MVT_TIME_UTC_mvt - EOBT_1_flt`  (take-off time minus the **estimated**
  off-block time from NM flight-plan update messages)
- `sched_delay = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` (take-off minus **schedule**)

Target: `offset = y - anchor`, i.e. `y - (MVT_TIME - EOBT_1)`. Since `y = MVT - BLOCK`
exactly (their T1), the offset is algebraically **`EOBT_1 - BLOCK`** on rows with an EOBT,
and **`SCHED - BLOCK`** on rows without one. Their own docstring states the intent
[S, `train_v54_base.py:1-12`]: *"The offset has a tight distribution (5th to 95th
percentile ~-1500 to +250 s)... A tree on the offset does not need a linear leaf to
reproduce the `y = sd` slope on fallback rows; it just needs to predict 0."*

Serving [S, `src_v3/predict_v54.py`]: score the three boosters directly, **bypassing
`predict_v30`'s clip at 0** (which would destroy negative offsets), add the anchor per row,
then clip to `anchor + [q0.001, q0.999]` per airport from `models/v54_offset_bounds.json`,
then clip at 0, then MS1 + MS2. LIRF rows taken unchanged from the v51 parquet.

### 3.2 Which rows were bimodal, and why anchoring failed on them

[S, `RECAP.md:72`]: *"Failed because anchor is bimodal on 2026 delayed rows: rows with
`mvt_eobt1 > 10,000 s` (aircraft held past EOBT) but normal taxi output
`pred_y = anchor + small_offset = 10,000+ s` instead of the true 1,000 s. 101 rows at EHAM
over 7,200 (v51 had ~4), 25 at LFPG (v51 ~4). The tree learns median offset ~-300 s, cannot
correct on huge-anchor rows."*

Mechanism, restated precisely [X — my reading of their code and numbers, consistent with
their own text]:

1. `mvt_eobt1` is **not** a measurement of off-block. It is `MVT_TIME - EOBT_1`, and
   `EOBT_1` is a *plan estimate issued before the fact*. When the flight is held, the
   aircraft still pushes back near its real off-block time but takes off hours after the
   EOBT it was issued. `mvt_eobt1` then balloons to 10,000+ s while the true taxi-out `y`
   stays at ~1,000 s.
2. The offset `EOBT_1 - BLOCK` therefore has **two modes**: a tight mode near -300 s on
   on-plan rows (the 5th-95th range they quote, -1500 to +250) and a long left tail
   reaching -8,000 to -9,500 s on held rows. Their own per-airport q0.001 bounds show the
   tail: EDDF -8,248, EHAM -9,509, LEBL -9,422, LFPG -9,476
   (`models/v54_offset_bounds.json`).
3. A squared-error tree on a bimodal target with a small minority mode **predicts the
   dominant mode**. The learned offset is ~-300 s everywhere the features do not
   separate the modes; nothing in the 117-column feature set reliably says "this aircraft
   is being held right now" ahead of take-off.
4. Reconstruction then multiplies the error by the anchor: `pred = anchor + (-300)`. On a
   row with `anchor = 10,000` the served value is ~9,700 against a true 1,000 — a
   **~8,700 s** single-row error, ~75,000,000 s² on one row.
5. **Their safety net could not fire on precisely these rows.** MS1's per-airport upper
   bound at EGLL/EHAM/LFPG/LSZH/LTFM is waived when
   `flt_null == 0 AND mvt_eobt1 > 5400` [S, `src_v3/postprocess.py:62-70`,
   `MS1_MVT_EOBT1_MIN = 5400`]. That exemption was written for the raw-`y` base, where a
   large `mvt_eobt1` was *evidence of a genuine hold*. Under MB1 a large `mvt_eobt1` is
   *mechanically the cause* of the inflated prediction. The guard and the bug select the
   same rows, so the guard let every one of them through. Evidence: `over_7200` went
   108 -> **279** with `at_zero_clip` 0 [S, `submission/kind-mango_v54.check.json`].
6. Aggregate footprint [S, same file]: 317,889 of 344,841 rows moved; per-airport mean
   shift **all upward**, EHAM **+25.54 s**, LSZH +10.67, LFPG +7.60, EDDM +6.00,
   EDDF +5.89, LEBL +4.88, EGLL +3.68, LEMD +3.46, LTFM +2.49; max |diff| 6,300-7,025 s at
   six airports. The mean shift is small but the tail is fat — exactly the RMSE-hostile
   shape.

[S, `RECAP.md:72`] Their own post-mortem also concedes the audit process failed here:
*"Grade D audit missed the bimodality. Lever closed."* MB1 had been ranked a
high-value measure (`MODEL_ANALYSIS.md:892-897`, fixing finding M1, *"plausible range
500 to 3,000 MSE"* of gain).

### 3.3 How their anchor differs from `MVT_TIME - AOBT_3` — and is the failure about
anchoring, or about *this* anchor?

**Verdict: it is about the choice of anchor, not about anchoring as such.** [X — my
analysis; their repo never makes this comparison because they refuse to touch `AOBT_3`.]

The two constructions are structurally identical in form, `y ≈ (MVT_TIME - X) + offset`,
and differ entirely in what `X` is:

| | kind-mango v54 | merry-quicksand (per slot's other investigator) |
|---|---|---|
| anchor | `MVT_TIME - EOBT_1_flt` (fallback `MVT_TIME - SCHED`) | `MVT_TIME - AOBT_3_flt` |
| what `X` is | **estimated** off-block, issued *ex ante* from NM flight-plan messages | **actual** off-block time, a post-hoc reported time |
| relation to label | `offset = EOBT_1 - BLOCK`: a *plan-vs-actual* error, unbounded, regime-dependent | `proxy - y = AOBT_3 - BLOCK`: a *reporting-offset* between two measurements of the same physical event |
| distribution | **bimodal**: tight near -300 s, long tail to -9,500 s on held rows | near-degenerate; kind-mango themselves measure `AOBT_3` within 60 s of off-block on **17-24 % of 2025 rows** at nine airports (7.9 % at LTFM) [S, `MODEL_ANALYSIS.md:784-787`] and state `MVT_TIME - AOBT_3_flt` *"reconstructs the hidden off-block time to within the reporting-offset noise"* [S, `README.md:36-38`] |
| failure mode when the anchor is wrong | error is **the size of the hold** (hours) | error is the size of the reporting offset (seconds to minutes) |

So the v54 result is evidence that **anchoring on a planned/estimated time fails when the
plan-vs-actual gap is itself the thing that has a fat, poorly-predicted tail**. It is *not*
evidence that residual-against-a-proxy learning fails. In fact the reverse: the closer the
anchor is to a true measurement of off-block, the *smaller and more unimodal* the residual,
and the better the reparameterisation works. `AOBT_3` is a measurement; `EOBT_1` is a
forecast. kind-mango anchored on a forecast and paid for the forecast's tail.

Two corroborating facts from their own repo:

- They independently estimate that a model reading `AOBT_3_flt` would land at
  **258-270 s** live [S, `MODEL_ANALYSIS.md:788-792`], which brackets merry-quicksand's
  reported 257.55. Their conclusion [claim]: *"This does not prove what other teams did."*
  It does, however, mean the two reports are not in contradiction — the anchored-offset
  *family* works, and the specific anchor decides whether it works.
- Their remedial measure MH7 [S, `MODEL_ANALYSIS.md:955-958`] is explicitly a hold rule:
  *"Raise the prediction toward `mvt_eobt1` when `mvt_eobt1 > 7,200` and a record exists;
  drop the rule if the anchored target already tracks the holds on the folds."* They
  understood the failure is the hold regime, not the parameterisation. They closed the
  lever before testing a regime-aware version.

Residual risk we must keep in view [X]: MB1 also removed `linear_tree`'s job (the offset
target was meant to make linear leaves unnecessary) while keeping the retuned linear-leaf
hyperparameters, and it was stacked on MB3 rather than run against a same-run control.
Their own calibration table (§1.4) says stacked changes mis-price. So +35.86 is an upper
bound on the harm of the anchor choice, not a clean isolation of it.

---

## 4. Mechanisms for the other three items

### 4.1 Why dropping `ec_*`/`opdi_*` gained 13.14 s (not 26.41): **zero-imputation
routing, not leakage and not misjoin**

[S, `RECAP.md:222-226`]:
- *"`ec_*` features have 0 % coverage in July 2026 (55 % of the scoring set); model routed
  all July rows to NaN branch, poisoning predictions."*
- *"v25 (impute with 2025 medians) failed catastrophically at 375.33 (+58 s worse); proves
  `ec_*` are actively harmful, not just missing."*
- *"`opdi_*` dropped too: dead weight (+0.09 CLEAN removal cost), and coverage collapses at
  5 airports in 2026."*
- *"Hold-out full 392.33 (v24 was 396.75, -4 s). Doc predicted only -1.9 s from the `ec_*`
  fix but LIVE gain was -13 s!"*

Precise mechanism, from their D1 finding [S, `MODEL_ANALYSIS.md:414-437`]:

> *"LightGBM's `NumericalDecision` (`include/LightGBM/tree.h`) converts NaN to 0.0 when the
> split's feature had no missing value in training."* ... *"on every July 2026 LIRF row the
> gate and `R_norm` read zero daily traffic and zero pre-departure delay, below any
> training value. Split tests route the row to the lowest-traffic branch; linear leaves
> extrapolate. The 2025 hold-out has full coverage and cannot see it."*

So: **not leaking, not misjoined, not merely noisy.** The failure is *out-of-support
missingness at serve time on a feature that was 100 % present at train time*. Because the
training data had no NaN in those columns, LightGBM's NaN→0 convention sends every 2026 row
down the "zero traffic, zero delay" branch — a branch whose leaf was fitted on the quietest
days of 2025 — and the linear leaf then extrapolates from there. Median imputation (v25)
made it worse because it moved every July row to one *dense, confidently-fitted* branch
instead of an edge branch, so the damage stopped being random and became systematic
(+58 s).

Note the coverage numbers [S, `MODEL_ANALYSIS.md:1143-1145`]: 7 same-day `ec_*` columns go
100.0 % -> **0.0 %** non-null at LIRF in July; their 7 lag-1 copies go 99.8 % -> **3.2 %**.
Jan-2026 was still 100 %. This is a mid-scoring-period feed outage in an external dataset.

**Their conclusion generalises to a rule, not to "don't use external data"** [X]: the
defect is *a feature whose 2026 coverage differs from its 2025 coverage*, and they encode
that as measure MD2 [S, `MODEL_ANALYSIS.md:857-862`]: *"A column whose coverage in either
2026 month fails the MP8 rule against the same 2025 month leaves every model."* They
continued to ship METAR, OSM, VRS and (via `plan_taxi_res`) route history throughout. The
13.14 s is a coverage-drift tax on two specific feeds, not a verdict on external data.

Corroborating: they later tried the same removal in the LIRF head (v49), where the paired
gain was **-2,405 MSE**, and live went **+0.42 s** — the gain did not transfer, and they
wrote [S, `RECAP.md:76`]: *"The v41 head's zero-imputed `ec_*` rows in July 2026 were
apparently routed to sensible branches, not the 'wrong' one D1 predicted."* So even their
own mechanism story is only half-confirmed: it held for the base, not for the head.

### 4.2 What "training on served rows only" (v51, MB3) means precisely

[S, `src_v3/train_v51_base.py:1-14` and the code]:

```python
keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
```

The serving stack overwrites **every** LIRF row with the LIRF regime head, then Step A on
top (`src/predict_v30.py:197-215`). So the base is never asked to predict a LIRF row —
yet it was being *trained* on 160,704 of them. Those rows are pathological: **16.6 % of
LIRF rows are fallback** (`BLOCK_TIME` within 60 s of schedule — a reporting artefact where
`y ≈ sched_delay`), and **12 of the 14 24-hour rows in all of 2025 sit at LIRF**.

Mechanism [S, `MODEL_ANALYSIS.md:567-576`, finding M3]: *"rows whose label the served value
never reproduces shape splits and leaf slopes that served rows pass through."* With
`linear_tree=True` and `min_data_in_leaf` 291, a handful of `y = 86,400 s` rows rotate a
leaf's linear slope, and every served row routed to that leaf is extrapolated along it.

Delta and its character [S, `RECAP.md:74`]: live **-5.99 s vs v48** (and -11.06 vs v41).
Crucially: *"Live -5.99 s at zero paired price — the paired hold-out cannot see this
because the base's errors on 2025 rows are moderate; MB3's benefit shows up entirely on
2026 OOD rows."* Secondary evidence that the mechanism is the one claimed: post-processing
MS1+MS2 moved only **85** rows with max 304 s under MB3, against 109 rows and max
**13,966 s** under v48 — *"the MB3 base already avoids the extreme predictions natively."*

(My brief said "-8.32". The repo says -5.99 vs v48 / -11.06 vs v41. No -8.32 anywhere.)

### 4.3 What "broke train/serve alignment" means (v60, MD4, +2.25 s)

[S, `src_v3/train_v60_base.py:1-21`] MD4's construction: for each key `K` in
`(op_apt, op_apt_rwy, op_apt_hbin, op, op_apt_stand)` and each month `m` in 1..12, build
the encoder on *all months except m* and assign that value to training rows in month `m`.
At serving, apply a single **full-year** encoder map.

[S, `RECAP.md:66`] verbatim: *"Two mechanisms cooperated: (1) the 'leak' MD4 was meant to
fix is small (1/N per row where N>=20), and removing it adds noise without removing
overfitting; (2) MD4 breaks the training-serving alignment (training rows use per-month
fits missing their own month, serving rows use the full-year fit), so the tree learns split
thresholds calibrated to the OOF distribution and misapplies them at serve time. LTFM shift
+11.73 s was the biggest single-airport move of any encoder lever."*

So "alignment" here means **the numerical distribution of the encoder column differs
between fit time and serve time**. An 11-month-out-of-12 median is noisier and slightly
shifted relative to a 12-month median; LightGBM's split thresholds are learned on the
noisier distribution and then applied to the tighter one, so rows land on the wrong side of
thresholds. This is the standard OOF-target-encoding hazard, and it is *not* fixed by
"being more honest" — the honest version is a different column from the one you serve.

**The pair of results is the interesting part** [S]: v58 went the *opposite* direction
(refit encoders on all 12 months, i.e. more in-sample leak) and lost **+1.29 s**; v60 (OOF)
lost **+2.25 s**. Their conclusion: *"v57's v26 encoders (partial exclusion of months 1,7;
training-serving aligned) sit at the local optimum. Both encoder levers now closed."* The
surviving configuration is the one where train and serve read the *same* map, even though
that map is partly in-sample.

The leak itself, quantified [S, `MODEL_ANALYSIS.md:339-352`, finding L1]: each training
row's own label sits inside its own key median and std at weight ≥ 1/20 (`MIN_COUNT = 20`);
*"One 24-h row sets a key's standard deviation near 10,000 on the row that caused it."*
Their MD4 spec (never shipped) also said to drop the std column entirely.

### 4.4 The two classifier-overfit failures

[S, `RECAP.md:193,196`]:
- **Step 3 fallback-and-24h classifier**: hold-out **-40**, live **+195**. *"Classifier
  does not generalize to 2026."* Deployed as a hard swap (`P(y>3600)` classifier + swap);
  `README.md:376-378` adds: *"live went from 372 to 625. Hard swaps have unbounded
  downside."*
- **Step C LIRF+EGLL fallback-detector soft mix (v17)**: hold-out **-8**, live **+5**.
  *"Classifier overfits 2025 hold-out."*
- Their later framing [S, `README.md:400-403`]: with the corrected fallback definition
  (`|y - sd| <= 1` at nine airports, `< 60` only at LIRF), *"these detectors model
  punctuality, not reporting failure. The `R_all` regressor already holds that signal."*

Mechanism [X]: the detector's positive class at nine airports is not a distinct physical
regime — it is the tail of the punctuality distribution, which the regressor already fits.
The classifier therefore learns 2025's punctuality level, which drifts. At LIRF the
positive class *is* a distinct artefact (block time pinned to schedule), and there the same
architecture works (-9.8 s at v22).

---

## 5. LIRF: what regime, what head, did they build it

**Yes, they built it, and it is their second-largest win after the three-bug fix.**

**The regime** [S]. LIRF has a reporting artefact: a large share of rows where
`BLOCK_TIME_UTC_mvt` is pinned to the schedule, so the published `y` equals `sched_delay`
rather than a real taxi-out. **16.6 % of the 160,704 LIRF training rows are fallback**
(`|y - sd| < 60`) [S, `train_v51_base.py:1-11`, `MODEL_ANALYSIS.md:571`], and **12 of the
14 24-hour rows in the whole 2025 training set sit at LIRF** — plus a separate `sd > 14,400`
null-flight-record class. Non-LIRF airports have a 3.2-5.8 % fallback share (EDDM, EGLL,
LEBL, LEMD, LFPG, LTFM) and **no head** — their open finding T5, ceiling 1,726 MSE.

**The head, exactly as served** [S, `README.md:70-84`, `MODEL_ANALYSIS.md:63-75`]:

| LIRF row class | served value |
|---|---|
| null `FLIGHT_ID_mvt` and `sd > 14,400` | Step A band table: `p_fb * sd + p_24h * (86,400 + mean_24h_extra) + p_norm * mixture`, 11 bands, mutually exclusive since v30 |
| `sd > 70,000`, outside Step A | ITY340 rule: `(5/6) * (86,400 + 1,150) + (1/6) * sd` |
| all other LIRF rows | `mixture = p_fb_cal * sd + (1 - p_fb_cal) * R_norm_LIRF` |
| (every non-LIRF airport) | 3-seed base mean |

Components: `R_norm_LIRF` = 5-member LightGBM `linear_tree` mean trained **only on LIRF
"genuine" rows** (`|y - sd| >= 60` and `y < 80,000`), 127 leaves, 166 columns, LIRF-only
encoders. `p_fb_LIRF` = LightGBM binary classifier for `|y - sd| < 60` on all LIRF rows,
167 columns including 14 leave-one-month-out fallback-rate encodings, isotonic-calibrated.

**What it was worth** [S]: `R_norm` alone cut LIRF clean RMSE **565.10 -> 318.08 (-247 s)**
(`RECAP.md:243`); the head shipped at v22 for **-9.8 s live**, plus v23 rate encodings
-2.8 s, Step A -58 s, the 6.3 rule -1.77 s, v30 LIRF-only encoders -1.28 s, v41 -0.65 s,
v55 -0.42 s, v56 -0.33 s. LIRF still holds **21 % of the v41 hold-out MSE**
(`MODEL_ANALYSIS.md:612`) and 13,984 of the 60,365 clean-class MSE.

**Their warning about the naive version** [S, `RECAP.md:249`]: *"Naive v19 mixture (uniform
mixture everywhere) FAILED at LIRF (+5.3G MSE) because `R_norm` underestimates on high-`sd`
rows and soft mix pulls all rows toward `sd`."* And their open defect T1: Step A reads the
*mixture* as its normal term instead of the genuine-row regressor, so the schedule weight
is counted twice (`p_band + p_norm × p_gate`).

---

## 6. Data drift 2026 vs 2025 — their measurements and their methods

We have measured none of this and our scored set is a different year from our training
data, so the *methods* here are the transferable part.

### 6.1 Method 1 — column-coverage monitor (label-free, cheap, runs before every upload)

[S, `src/check_coverage.py`]. Per served column, per airport, per period, compute the
**non-null share** and the **non-zero share** on three slices: full 2025, January 2026,
July 2026. Fail the build when:
- non-null share drops by more than **20 points** vs full 2025, **or**
- non-zero share drops by more than 20 points vs full 2025 **and** vs the *same month* of
  2025 (the month match absorbs seasonality — a good detail).

Output `models/coverage_monitor.json`; exit 1 on a strict failure. 110 base columns are
strict; LIRF-only columns are informational.

**Strict failures they got** [S, `coverage_monitor.json:failures`], all 4:
```
dewpt_spread at LEMD in 2026_jan: non-zero share dropped 24 points
dewpt_spread at LSZH in 2026_jan: non-zero share dropped 22 points
eobt1_sched at LTFM in 2026_jul: non-zero share dropped 21 points
eobt1_sched_slog at LTFM in 2026_jul: non-zero share dropped 21 points
```
Their verdict [S, `RECAP.md:26-30`]: *"4 strict cells flagged, all data drift, not pipeline
defects."*

### 6.2 The drift table, verbatim [S, `MODEL_ANALYSIS.md:1143-1161`, Appendix A1]

Shares in percent. "Non-zero" = share of rows with a non-zero value.

| column | measure | airport, period | 2025 | 2026 |
|---|---|---|---|---|
| 7 same-day `ec_*` in the LIRF head | non-null | LIRF, July | 100.0 | **0.0** |
| 7 lag-1 `ec_*` in the LIRF head | non-null | LIRF, July | 99.8 | **3.2** |
| `flt_null` | non-zero | EHAM, January | 1.6 | **5.0** |
| `flt_null` | non-zero | LSZH, Jan / Jul | 1.5 | **3.0 / 2.7** |
| `flt_null` | non-zero | LIRF, July | 0.9 | 1.7 |
| `flt_null` | non-zero | LEBL, July | 1.0 | 2.0 |
| `flt_null` | non-zero | LEMD, July | 0.4 | 0.9 |
| `flt_null` | non-zero | EDDF, January | 0.8 | 1.4 |
| `eobt1_sched` | non-zero | **LTFM, July** | 40.3 | **19.6** |
| `eobt1_sched` | non-zero | **LFPG, July** | 35.7 | **61.6** |
| `eobt1_sched` | non-zero | EGLL, July | 46.5 | 59.3 |
| `eobt1_sched` | non-zero | LFPG, January | 35.7 | 44.8 |
| `arr_same_rwy_prev_15m` | non-zero | **LFPG, Jan / Jul** | 19.9 | **3.9 / 3.8** |
| `arr_same_rwy_next_10m` | non-zero | LFPG, Jan / Jul | 17.6 | 2.9 / 2.6 |
| `openc_op_apt_stand_median` | non-null | EDDF, July | 92.0 | 74.0 |
| `openc_op_apt_stand_median` | non-null | EDDM, July | 90.2 | 80.4 |
| `openc_op_apt_stand_median` | non-null | LSZH, January | 88.3 | 78.8 |
| `ground_time` | non-null | EDDF/EDDM/LEBL, Jan | 94.5/92.4/95.3 | 90.4/87.5/91.0 |
| `stand_gap` | non-null | LIRF, January | 4.0 | 1.8 |
| `ades_arr_atfm_delay_today` | non-null | all, July | 4.2-10.0 | 4.0-9.5 |

Notable, with their reading:
- **LEMD / LSZH**: the January drift is `dewpt_spread` reading zeros — a METAR
  dewpoint-field artefact, plus the LSZH stand-encoder coverage loss (88.3 -> 78.8) and
  `flt_null` doubling (1.5 -> 3.0).
- **LTFM**: `eobt1_sched` non-zero halves, 40.3 -> 19.6 %. Their D4 reading: *"the
  information content of `mvt_eobt1` changes by airport and month"* — in July 2026 four
  fifths of LTFM rows have `EOBT_1` equal to the schedule, so the rank-1 feature carries
  nothing there. LTFM is **13.8 %** of the scoring rows.
- **LFPG**: opposite direction, 35.7 -> 61.6 %, *and* the `mvt_eobt1` median moved
  1,441 s (2025 months 1 and 7) -> **1,143 s** (2026) — a 298 s shift in the
  highest-gain feature. LFPG is **11.6 %** of scoring rows. Separately its same-runway
  arrival counts collapsed by 16 points (under the 20-point rule, so the monitor **passed**
  them — their D5; likely a runway-identifier change).
- They log all of this as **unhandled**: cause #8 of their 10 causes is *"2026 drift is
  logged, not handled"* [S, `MODEL_ANALYSIS.md:170-176`].

### 6.3 The "25 % LSZH" number — what it actually is [S]

`models/coverage_monitor.json:opdi_taxi_out_counts` — OPDI `taxi_out_v2` **record counts**
per airport-month, not traffic:

| airport | 2025 monthly mean | Jan-2025 | Jul-2025 | Jan-2026 | Jul-2026 | within 20 % (month-matched) |
|---|---|---|---|---|---|---|
| LSZH | 9,437.7 | 7,542 | 11,770 | **5,630** | 12,588 | false |
| EDDF | 2,418.4 | 7 | 615 | 5,582 | 11,388 | true |
| EDDM | 1,046.5 | 1,762 | 1,547 | **1** | 995 | false |
| LEBL | 1,938.9 | 6,912 | 458 | **6** | 58 | false |
| EHAM | 109.3 | 174 | 67 | 10 | 65 | false |
| LEMD | 1,796.9 | 0 | 1 | 1 | 64 | false |
| LFPG | 112.5 | 120 | 0 | 0 | 3 | false |
| EGLL | 7.1 | 15 | 3 | 2 | 15 | false |
| LIRF | 2.2 | 0 | 0 | 0 | 3 | false |

LSZH Jan: 5,630 / 7,542 = **-25.4 %**. Their reading [S, `RECAP.md:30`]: *"the L6 January
gate fails as written. July 2026 (12,588 vs 11,770) passes."* The whole table shows OPDI
coverage is erratic-to-absent at 8 of 10 airports in both years — which is why `opdi_*` was
dropped and why they only ever recommended OPDI raw counts *"at EDDF and LSZH only"*
[S, `README.md:388-390`].

### 6.4 Method 2 — the **arrival mirror** (label-bearing 2026 validation). The best idea in
the repo.

[S, `MODEL_ANALYSIS.md:314-323`, finding P11, graded **V, high**]:

> *"344,693 arrival rows of 2026 carry in-block time and taxi-in."* ... *"a taxi-in model on
> the same code path, trained on 2025 arrivals and scored on 2026 arrivals, measures the
> 2025-to-2026 transfer of every lever with real labels. The team spent upload slots
> instead."*

First use as a bare drift meter [S, `RECAP.md:214`]: 2025 hold-out **202.40 s** -> 2026
**214.96 s**, **+6.21 %**. Per airport: EHAM **+128.9**, LIRF **+28.9**, LFPG **+24.1**;
EGLL **-40.9**, EDDF -13.3, LEBL -27.6. So drift is **large, airport-specific, and
signed both ways** — not a global shift a single calibration could absorb.

Built out as `src_v3/arrival_mirror.py` (MP6), deliberately minimal: features are airport,
`ADES`, runway, stand, stand-prefix, aircraft type, operator, flight prefix, hour, dow,
month, arrival schedule delay. Results [S, `models/v3/arrival_mirror.json`]:

- full 2026 RMSE **199.14** on 344,565 test rows (trained on 2,082,048 2025 arrivals)
- per month: **January 222.14**, **July 178.71**
- per airport: EHAM 282.21, EGLL 281.81, LIRF 260.00, LFPG 219.27, LTFM 175.84,
  EDDF 157.32, LEMD 122.47, LSZH 114.56, EDDM 114.25, LEBL 75.38
- per airport-month, the standout: **EHAM January 405.53 vs EHAM July 129.38** — a 3.1×
  swing within the same scoring set, matching the EHAM `flt_null` jump 1.6 -> 5.0 % in
  January.

Their acceptance rule [S, `MODEL_ANALYSIS.md:827-830`, MP6]: *"A lever that helps the 2025
folds and hurts 2026 arrivals does not ship."*

**They never used it as the gate.** It was built after most of the ledger and only MF4
(v59) was justified by it — and MF4 lost +0.37 s, with their own post-mortem [S,
`RECAP.md:67`]: *"drift signal from the arrival side does not transfer to departure taxi-out
in the way the arrival mirror suggested."* So: the mirror is credible as a **drift
measurement instrument**; it is **unvalidated as a feature source**, and unvalidated as a
gate because they never ran it as one.

### 6.5 Method 3 — a labelled-2026 rolling residual (MF4/v59), and why it failed

[S, `src_v3/build_arr_drift.py`, `RECAP.md:67`]: four columns = 1-day and 7-day rolling
median of arrival taxi-in minus the 2025 fit-month per-airport median, keyed on
`MVT_ID_mvt`, coverage 0.998 train / 0.982 rank. Base retrained at 121 features. Non-LIRF
mean shifts -0.73 to +1.71 s, biggest at LTFM and EGLL (the largest arrival-side drift).
Live **+0.37 s**. Their two candidate mechanisms [claim]: *"(1) 2026 departures may already
be capturing the drift via existing features (congestion, tempo); (2) rolling arrival
medians add noise on quiet days that outweighs any signal on busy days."*

---

## 7. The `AOBT_3_flt` / `LOBT_flt` exclusion

**Decision** [S, `README.md:32-58`]: permanent exclusion of both fields from every
submission. Verbatim:

> *"The challenge brief forbids 'exploiting the ranking process'. We stay off two inputs
> that would place us in the top 3 but that we read as excluded:*
> 1. *`AOBT_3_flt` on ranking DEP rows — present on 98.9 % of ranking rows;
>    `MVT_TIME - AOBT_3_flt` reconstructs the hidden off-block time to within the
>    reporting-offset noise. The brief warns 'using them is probably exploiting the ranking
>    process'.*
> 2. *`LOBT_flt` — equals `AOBT_3_flt` on 4.3 % of rows and is otherwise a planned time,
>    not an actual off-block. Excluded for the same reason.*
>
> *Both fields are visible in the training and ranking files. Our submissions never read
> them. The exclusion sets the floor near 300 s on the leaderboard."*

**The quantified cost of the stance** [S, `MODEL_ANALYSIS.md:781-797`, finding X1, graded
"V, decision"]:
- `AOBT_3_flt` lies within 60 s of off-block on **17-24 % of 2025 rows** at nine airports,
  **7.9 % at LTFM**; `LOBT_flt` equals it on 4.3 %.
- *"if 17 to 24 % of rows became near-exact and those rows carried the average error, the
  live MSE falls to 66,850 to 73,000, or **258 to 270 s**. The top score, 263.46 s, sits
  inside that range. This does not prove what other teams did."*

**The organiser rulings they logged** [S, `RECAP.md:335-379`]:
- 2026-09-08 (espinielli): *"Organiser is not providing any value as ground truth for
  submissions. Any provided value is fair to use as a proxy (or as itself) if it helps."*
- 2026-09-11: *"No phase 2 is planned"*, though reserved *"if reverse engineering the
  ranking is too easy — which we despise."*
- 2026-09-18 (MX1, resolved): espinielli, asked directly about `AOBT_3_flt`/`MVT_TIME` and
  ADS-B-derived off-block times: *"the model is for post-ops, not for tactical use"* and
  *"there are no such restrictions: if you can find open trajectory data with good ground
  coverage and extract off-block times, we are ok. Practically speaking it won't be
  possible"* (poor coverage at LFPG, LIRF, LEMD, LTFM given as the reason).
- Their response [S, `README.md:47-58`]: *"No formal rule bans either input. We keep the
  exclusion anyway: the organiser separately called ranking reverse-engineering something
  'we despise', and their own answer relies on poor ADS-B surface coverage at most airports
  to make the exploit impractical, not on a rule against it."*
- Also logged: **Trino data access is NOT allowed** (confirmed by espinielli); EUROCONTROL
  AIU daily files and ADSB.lol (ODbL 1.0) **do** count as open data if declared with
  attribution; open data generally allowed if declared.

**Reading for us** [X]: the organiser has now stated on the record that no rule bans
`AOBT_3_flt`. kind-mango's exclusion is an ethical choice they document as costing them
~30-40 s of leaderboard position. It is not a compliance finding, and it should not be
imported as one. It does, however, mean that any public entry near 257-263 s should be
assumed to be reading `AOBT_3_flt` unless it demonstrates otherwise — which bears directly
on how we interpret merry-quicksand's `MVT_TIME - AOBT_3` proxy.

---

## 8. Reliability assessment of this source

**Strong.** The repo is unusual in that its ledger records failures with the same precision
as wins, cites file paths and line numbers throughout, records predicted-vs-landed for
several ships (v33: predicted 301.86, landed 301.87), and contains a self-audit that grades
its own evidence (V/R, high/medium/low, grades C/D for cost estimates). Their `MODEL_ANALYSIS.md`
explicitly lists corrections to its own earlier passes.

**Caveats.** (a) Three numbers in my tasking do not match the repo (§0) — treat second-hand
summaries of this repo with suspicion and go to the file. (b) Several deltas are stacked,
not isolated: v44 and v45 were **never scored**, so the base-stack attribution is
uncertain by their own admission (finding P6). (c) Their paired hold-out is demonstrably
mis-calibrated for stacked changes (§1.4) — a "gate failed by 7 MSE" verdict (L7 weather)
is not evidence the lever is dead. (d) LIRF hold-out numbers are in-sample (their L2), so
LIRF-head deltas measured offline are optimistic. (e) 287.33 is a minimum over ~35 uploads;
they estimate **200-400 MSE of their own progress since v30 is selection on the scoring
set** (finding P12).

---

## Applicable to resilient-hedgehog

**1. The anchored-offset failure does NOT generalise to our LIRF reparameterisation — but
it names the exact condition under which ours would fail.** [X, from S]

kind-mango's +35.86 s is a failure of *one anchor*, not of anchoring. Their anchor was
`MVT_TIME - EOBT_1`, where `EOBT_1` is an **ex-ante plan estimate**; the residual
`EOBT_1 - BLOCK` is bimodal because held aircraft blow the plan by hours while taxiing
normally. merry-quicksand's anchor is `MVT_TIME - AOBT_3`, where `AOBT_3` is a **reported
measurement of the same physical event as the label**; that residual is a reporting offset
of seconds. Same algebra, opposite residual distributions. The mechanism is:

> An anchored-offset target is safe iff the residual `anchor_time - true_off_block` is
> unimodal and small relative to `y`. It is dangerous iff the anchor can be wrong by an
> amount much larger than `y`, because the reconstruction step `pred = anchor + offset`
> multiplies a small conditional-mean error by an unbounded anchor.

**So the actionable test for our reparameterisation is: plot the distribution of
`our_anchor - true_off_block` on training data, per airport and per regime, and check for a
second mode.** If our LIRF anchor is derived from a *planned* or *estimated* time, we are in
kind-mango's failure regime and should expect it to blow up on 2026 held rows regardless of
how well it fits 2025. If it is derived from an *observed* time, we are in
merry-quicksand's regime. This is a one-hour check and it settles the generalisation
question empirically rather than by analogy.

**2. Whatever guard we have against extreme predictions, verify it is not keyed on the same
quantity that causes the blow-up.** kind-mango's MS1 cap exempted rows with
`mvt_eobt1 > 5400` — precisely the rows MB1 inflated
(`src_v3/postprocess.py:62-70`). The guard and the bug selected the same rows, so
`over_7200` went 108 -> 279 with the guard active. If we generalise an anchored target, any
clip conditioned on "large anchor = genuine hold" becomes a clip conditioned on "large
anchor = large error" and silently inverts. Re-derive guards **after** a target change,
never inherit them.

**3. Conversely: MH7 is the version they never tried, and it is probably the right one for
us.** [S, `MODEL_ANALYSIS.md:955-958`] They wrote the remedy — a hold-regime rule on top of
the anchored target — and closed the lever before testing it. A **regime-conditional**
anchored target (anchor only where the anchor is trustworthy; fall back to raw `y` where
the hold indicator fires) is untested public ground. Given that we already have a LIRF
regime head, this is a natural extension rather than a new architecture.

**4. On adding external data: their evidence says "gate on coverage parity", not "don't".**
The -13.14 s (not -26.41) came from two feeds whose 2026 coverage collapsed mid-scoring-
period (`ec_*`: 100 % -> 0 % non-null at LIRF in July; OPDI: erratic at 8 of 10 airports in
both years). The precise mechanism is LightGBM's **NaN→0 convention on features that had no
missing values in training**, which routes every serve-time row to a lowest-value branch
whose linear leaf then extrapolates. This is a *deployment* failure, not an information
failure — and median imputation made it four times worse (v25, +58 s). They kept METAR, OSM,
VRS and route history throughout. **The rule to import is: before adding any external
feature, measure its non-null and non-zero share on our scored period against the matched
period of our training year, per airport, and reject on a >20-point drop.** Their
`src/check_coverage.py` is ~90 lines and is directly portable.

**5. Measure drift with the arrival mirror — we have not measured drift at all, and they
have shown it is large, signed both ways, and airport-specific.** Arrival rows in the
scoring file carry real taxi-in labels (344,693 of them in their case). Train a taxi-in
model on our training-year arrivals, score it on our scored-year arrivals: 2025 hold-out
202.40 -> 2026 214.96, **+6.21 %**, but EHAM **+128.9** against EGLL **-40.9** — a global
calibration would be actively wrong at half the airports. Their built-out mirror shows
**EHAM January 405.53 vs EHAM July 129.38**. Caveat: they validated it as a *measurement*
instrument only; their one attempt to turn mirror-derived drift into *features* (MF4/v59)
lost +0.37 s. Use it as a gate, not as a feature source.

**6. Two of their specific drift findings are likely to affect us directly if our anchor is
EOBT-derived.** `eobt1_sched` non-zero share went 40.3 -> 19.6 % at **LTFM** (13.8 % of
scoring rows) and 35.7 -> 61.6 % at **LFPG** (11.6 %), and LFPG's `mvt_eobt1` median moved
**298 s**. The highest-gain feature in their model changes meaning by airport between years.
If our reparameterisation leans on EOBT, this is where it will decay first.

**7. Their paired-hold-out calibration table is a warning about our own offline gates.**
Stacked changes priced -1,864 MSE landed -106; an unpriced 12-month refit landed -1,530.
Single changes against a same-run control landed; stacks did not. If we are gating a
reparameterisation offline while other changes are in flight, the price is not
interpretable.

**8. Things they closed that are probably still open for us.** L7 weather joined at
**pushback time** (rather than take-off) failed their gate by **7 MSE** out of 300 while
improving all four served classes — that is a pass, not a fail, by any reasonable reading.
Their T5 finding leaves 1,726 MSE (2.9 s) of fallback error at the six non-LIRF airports
with a 3.2-5.8 % fallback spike and no regime head. And MB3 (train only on rows you
actually serve) was worth **-5.99 s live at zero offline price** — if any part of our stack
is trained on rows a downstream head overwrites, that is a free check.
