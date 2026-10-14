# Remaining competitor repos: deep read (2026-10-05)

All clones match the remote HEAD on 10-05 (`git ls-remote`). I read them for ideas only and copied no code. Each idea below was checked against `src/prc` (features.py, features_extra.py, levers.py, train.py), `configs/main_flseq_osm_nolirf.yaml` and `docs/`.

## Per repo

### merry-mushroom (#66, 276.03): TAUIL-Abd-Elilah/prc-data-challenge-2026
- **Licence and status:** GPL-3.0. Last commit 10-03. Official scores v3→v7: 289.2 → 288.5 → 283.0 → 280.5 → 278.8. v8–v10 were finalised locally but not uploaded.
- **Approach:** a v3 LightGBM base on the AOBT_3 residual, then fixed-weight "experts" applied in sequence.
  - Each expert's weight is picked from {0, .1, .25, .5, 1} on Jan/Jul and frozen.
  - It must then pass Nov/Dec with a positive paired UTC-day bootstrap.
  - It is then refit with Apr/Oct removed from both fitting and early stopping, and must pass there too.
  - Reserved guards follow on Feb/Aug and May/Sep.
- **Techniques:**
  - **Deep GPU CatBoost on the AOBT_3 residual** (`deep_timestamp_expert.py`, `deep_arrival_expert.py`, `traffic_deep_expert.py`): depth 9–10, learning rate 0.04, l2 12, random_strength 0.5, bagging_temperature 0.5, ctr 1, border 128, up to 10k trees, all of which are used. A 0.5 blend gave −2.1 s on Jan/Jul and −3.2 s on Nov/Dec. On Apr/Oct, depth 9 → depth 10 plus the ARR fields gave −4.2 s [3.9, 4.6]. That comparison is confounded by the added features. **Ours:** CatBoost depth 6, lr 0.2. **New: yes (capacity).**
  - **24 NM-neighbour features** (`proxy_neighbour_expert.py`): −1.27 s in a matched ablation. **Ours:** `nbr_*`, `fw_*` (L2) and `tx_*`. **Not new.**
  - **8 same-runway arrival headways** (`runway_arrival_features.py`). **Ours:** `seq_arr_*`. **Not new.**
  - **Interval-flow features (v11)** (`taxi_interval_flow_features.py`): takeoffs and pushes between own AOBT_3 and MVT, intervals active at takeoff, overtakes. Gains of −0.39 / −0.74 / −0.83 s. **Ours:** this is our L3 `qp_*`, rejected at +0.04, and the `overtaken_by_aobt` analysis. **Not new.**
  - **Same-runway ARR taxi-in context (v10)** (`runway_arrival_taxi_features.py`): count, mean and std of completed taxi-ins in 15/60/180 min, plus arrivals still taxiing, all per runway. Passed at weight 0.25: −0.18 s Jan/Jul, −0.50 s Nov/Dec. **Ours:** taxi-in mean at airport level only (`arr_taxi_in_mean_{30,60}m`, `bk_arr_*`). **New: partly.**
  - **Runway geometry (v12)** (`runway_geometry_features.py`, from OurAirports):
    - true heading, and head/tail/crosswind;
    - counts of intersecting runways and of near-parallel runways (≤15° and ≤2 km apart);
    - **ARR/DEP counts over the past 15/60 min on intersecting or parallel strips**.
    - Gains of −0.17 / −0.19 / −0.43 s (Apr/Oct) on top of v11.
    - **Ours:** headwind and crosswind (`weather.py`) and same-runway traffic only. **New: yes (other-strip traffic).**
  - **Schedule-source classifier at LIRF and LTFM** (`catboost_source.py`): matched rows with |AOBT_3 − SCHED| > 600 s get p·(MVT − SCHED) + (1 − p)·ŷ at half strength. Gains of −0.38 / −0.33 s, CI > 0. This overlaps with outgoing-boat's mixture, but adds LTFM and the half-strength shrink. **New: yes** (only the inventory lists it so far).
  - **Movement-only missing-clock model (v8)** (`movement_only_expert.py`): trained on all ~2.08M ordinary departures with labels in [0, 7200] and every NM field hidden. It replaces the prediction on non-LIRF rows with no NM record and gave −0.92 s (Jan/Jul) and −1.71 s (Nov/Dec) on all rows. **Ours:** the 09-16 `shared_b` test (all labels, all unmatched rows including LIRF, an immature base) was +35 on unmatched (non-significant) and rejected. **New: yes (this variant).**
  - **Event-sequence CNN for missing rows (v14):** 16 events before and 16 after. Not run. **New, but not worth it now.**
  - **Released fields not used by us:** ARVT_3 − AOBT_3, ARVT_3 − ARVT_1, and the full NM callsign as a category. Also second/minute "timestamp precision" features, which **break our jitter rule and must not be used.**
- **Negative results:**
  - Airport-specific blend weights reverse on the forward fold.
  - A historical flight analogue for missing NM rows gets worse forward.
  - A CPU CatBoost residual blend gets worse.
  - The LIRF long-schedule ratio, the all-airport missing mixture and the Rome clock classifier all fail forward.
  - Same-flight ARR landing minus ARVT_3 does not associate with the residual.
  - Recovering missing departure clocks from shared ARR records recovers 0 rows.
  - A 255-leaf LightGBM passed alone but failed in combination.
  - Their ID-order audit was rejected; that is our banned side channel.

### unique-umbrella (#98, 291.52): kagervig/PRC-Challenge-2026
- **Licence and status:** GPL-3.0. Last commit 10-01. LB: v27 313.0 → v34 305.0 → v36 293.9 → v39 291.0. A single LightGBM with 9 seeds.
- **Techniques:**
  - **Gate-hold backlog (v34)** (`model.py`, `learnings.md` §Phase 5): at own AOBT_3, count departures on the same runway with EOBT_1_j ≤ t < AOBT_3_j (planned off-block passed, not yet pushed; intervals capped at 2 h). Harness −1.30 s, LB −2.06 s. **Ours:** `bk_sched_backlog_60m` uses SCHED/MVT at takeoff and the bk group was rejected; L3 counts aircraft pushed but not airborne. The EOBT→AOBT gate queue is **new**. outgoing-boat's `ret_*` is a near-twin.
  - **Taxiway queue at pushback (v29):** −2.08 s LB. Equals our L3, which we rejected. **Not new.**
  - **EWMA congestion (v32):** EWMA of arrival taxi-in with a 10-min half-life beat a 60-min boxcar by 0.57 s. **Ours:** boxcar only. **New: minor.**
  - **LIRF no-NM override (v36/v39):** for rows with no NM record and MVT − SCHED ≥ 15,000 s, predict MVT − SCHED. LB −8.5 s, then −2.9 s. **Ours:** covered by the model-B SCHED reparametrisation.
- **Negative results:**
  - Target encoding on (airport, hour): harness −0.99 s, **LB +0.91 s**.
  - Runway-configuration counts: +0.4 to +0.65 s.
  - airport × stand IDs: +1.9 s.
  - log target +7.3 s and sqrt target +0.9 s.
  - Asymmetric loss and LIRF up-weighting: worse.
  - Day-level ATFM delay: correlation with the residual 0.02.
- **Heuristic:** congestion features gained 2–3× more on the LB than on their harness, but target-encoding gains reversed.

### joyous-rainbow (#117, 299.23): radekacar/joyous-rainbow
- **Licence and status:** GPL-3.0. Last commit 09-25. Score history: local 324.1 against LB 299.2.
- **Techniques:**
  - **Unmatched group (group A), a three-way average** (`taxiout_final.py`), RMSE within the group:
    - a per-airport cubic-plus-hinge-at-3 h curve on MVT − SCHED: 1,908;
    - a linear-tree LightGBM: 1,924;
    - a classifier mixture (AUC 0.956, p shrunk ×0.8): 1,860;
    - **all three averaged: 1,822**.
    - **Ours:** a single model B. **New: yes (diverse members in the unmatched lane).**
  - **Taxiway-graph Dijkstra distance from OSM:** LB +0.6. **Known** (kind-mango, enio).
  - **Visibility** carries the METAR signal: −4.5 s when it was accidentally dropped. **Ours:** METAR including visibility. Covered.
  - **linear_tree:** −31 LB for them. **Known** (P7, `inv_it11_lineartree`).
- **Negative results:**
  - Clipping, per-airport caps, Huber, shrinking toward typical values: all worse.
  - Tail classifier: +0.7 s.
  - Airport-normalised congestion: +0.5 s.
- **Their noise floor:** local changes below about 2 s did not reliably transfer to the LB.

### joyful-lemon (#116): fenil264/prc-2026-taxi-out
No licence file. Three commits, the last on 09-09. EDA notebooks only (EHAM has a runway spread of 4.4×; Spearman with distance is 0.30). **Nothing new.**

### sincere-donkey (#219, 625): GLAPM/prc-2026-sincere-donkey
GPL-3.0, one commit. An LGB + XGB + CatBoost blend on hour, day of week, month and categoricals, with categories unified across training and ranking. **Nothing new.**

### logical-ladder (#121): AbdullahKhan-x/PRC-2026
GPL-3.0. Download scripts only. **Nothing new.**

### kind-earthquake (#51): col3name/air-data
No licence file. HEAD is still `c14bea4` (09-05). **No change since our note 23.** Unrelated to ideas: the repo commits `credentials.json`.

## Ranked ideas new to us

| # | Idea | Expected value | Effort | Risk | How to measure |
|---|---|---|---|---|---|
| 1 | **Diverse members for the unmatched lane.** (a) An NM-blind direct model trained on all departures with labels in [0, 7200] and every NM field hidden, applied only to non-LIRF unmatched rows (merry-mushroom). (b) A per-airport MVT − SCHED cubic-plus-hinge curve (joyous-rainbow). Blend each with model B; pick the weight on jan/jul and freeze it for jul_fwd/forward. | High: the lane holds 24–39 % of MSE; merry-mushroom gained −0.9 to −1.7 s on all rows | 1 day | High variance on about 5k rows; EHAM rollovers. Never route LIRF through (a) | `prc.compare.verdict` on all 4 folds, plus the unmatched non-LIRF subset with K=200 drop and day blocks |
| 2 | **Deep CatBoost member:** depth 8–10, lr 0.04, l2 12, random_strength 0.5, up to 10k trees. Swap it for our depth-6, lr-0.2 member. | Medium-high: their deep model gained 2–3 s over depth 9 (confounded) | Compute-bound: hours on CPU, a GPU would help | Fit time with 6 days left | Rebuild v37 with the new member; paired CI on 4 folds; check that LB transfer is not like L5 |
| 3 | **Other-strip traffic:** ARR/DEP counts in the past 15/60 min on intersecting and near-parallel runways (≤15°, ≤2 km), from the OurAirports geometry we already hold. Matters at LSZH, EHAM, LEBL, LIRF, LEMD, EDDF 18. | Low-medium: −0.2 to −0.4 s for them | 0.5 day | Low; label-free | Anchored-LGB arm and verdict; per-airport split |
| 4 | **Schedule-source mixture on the matched lane at LIRF and LTFM:** when \|AOBT_3 − SCHED\| > 600 s, apply p·(MVT − SCHED) + (1 − p)·ŷ at half strength. Merge with the outgoing-boat idea already in the inventory. | Medium: −0.35 s, CI > 0 for them; outgoing-boat's version was larger | 0.5 day | p calibration in 2026 (LTFM EOBT==SCHED share shifts in July) | Post-hoc on OOF, so it is cheap: verdict on 4 folds, by airport and month |
| 5 | **Gate-hold backlog per runway:** #{j: EOBT_1_j ≤ AOBT_3_i < AOBT_3_j} on the same runway, also by airport. Null on unmatched rows. | Low-medium: −2 s LB on a weak model; our L3 relative was +0.04 | 2 h | Low | L-lever arm against v37 |
| 6 | **Per-runway ARR taxi-in context:** count, mean and **std** of completed taxi-ins in 15/60/180 min, plus arrivals still taxiing on that runway. | Low: −0.2 to −0.5 s for them | 2 h | Low | Same arm protocol; can be bundled with #5 |
| 7 | **EWMA (10-min half-life) variants** of the arrival taxi-in mean and of `nbr_mvt_minus_aobt`. | Low: about −0.5 s for them versus a boxcar | 2 h | Low | Same arm protocol |
| 8 | **Reserved-month guard for adoption:** refit with Apr/Oct (and Feb/Aug) removed from fitting and early stopping; require a positive day-block CI for fixed weights. | Process value: targets problem (e) | Compute only | None on the score | Backtest on L5 (CV −0.32, LB +0.12) and see whether the guard would have rejected it |
| 9 | **NM consistency features:** ARVT_3 − MVT (airborne time by mixed clocks) minus its route median; ARVT_3 − ARVT_1. These may flag clock desync. | Low | 1 h | Low | Gain ranking plus verdict |

**Do not adopt.** Timestamp-precision features (seconds and minutes) and ID-neighbourhood features break our rules. Target encoding on (airport, hour), runway-configuration categoricals, log/sqrt targets, asymmetric loss and airport-specific blend weights all hurt or reversed for at least two teams.
