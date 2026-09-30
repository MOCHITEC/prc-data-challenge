# Public code deep dive: merry-quicksand and kind-mango, 2026-09-30

This note records what the two strongest public codebases do, in enough detail to compare with our pipeline. It
covers methods, features, targets, data handling and published numbers. **No code was copied.** Both repos are
GPLv3, but the originality clause bars us from lifting another team's implementation, so this note describes and
does not transcribe. Nothing was fitted. `src/`, `configs/`, `tests/`, `outputs/` and `data/` were not modified.
Our own code was read at `HEAD` = `b5b0be7`. For `src/prc/train.py` I read `git show HEAD:` because the working
tree holds another agent's uncommitted edits.

Marks:
- **[S]** read from the cited file.
- **[X]** my inference or arithmetic.
- **[U]** unverified, or not established from any source I could read.

Slots 21 (`21-repo-merry-quicksand.md`) and 22 (`22-repo-kind-mango.md`) were read first and are not repeated
except where a line reference or a number is needed. This note adds commit and line references, the v10→v11
diff, the kind-mango v61–v89 track, and a feature-by-feature map to our columns.

---

## 0. Sources and freshness

| repo | team (board) | HEAD now | date | new since slot 25? |
|---|---|---|---|---|
| `saurabhdxt-labs/prc-challenge` | merry-quicksand (#24, 250.73) | `fdeb704` | 2026-09-12 16:34 −0400 | **no** [S] `git ls-remote`, fresh blobless clone |
| `skylinkapi/prc-data-challenge-2026-kind-mango` | kind-mango (#31, 264.34) | `2a4ec1b` | 2026-09-28 22:08 +0200 | **no** [S] |
| `col3name/air-data` | kind-earthquake | `c14bea4` | 2026-09-05 | no |
| `javidmardanov/PRC-Data-Challenge-2026` | elegant-alligator | `845d0d1` | 2026-09-15 | no |
| `Phoenix-Ops-LTD/prc2026-taxiout` | zestful-fountain | `4477d9b` | 2026-09-13 | no |
| `satam2/knowledgeable-helicopter` | knowledgeable-helicopter | `main` `9e5b9e6` (09-12); branch `experiments/280-screening` `e3d2339` | — | no (slot 25's `e3d2339` is the side branch) |
| `ChandanHegde07/OpenAir` | likable-eagle | `97f603c` | 2026-09-15 | no |
| `radekacar/joyous-rainbow` | joyous-rainbow (#100, 299.23) | `8dc92c0` | 2026-09-25 | read README only |

Access: `git clone --filter=blob:none` over the proxy worked for all eight. The clones are in the session
scratchpad under `competitors/`. **Everything below is at the HEADs above.** merry-quicksand's public code ends at
v12 = 266.37. Their path to 250.73 is not public.

---

## 1. merry-quicksand (`fdeb704`)

### 1.1 Target and lanes [S]

| lane | rows (scored) | target | reassembly | where |
|---|---|---|---|---|
| matched (`AOBT_3_flt` present) | 339,551 (98.466 %) | `delta = BLOCK − AOBT_3`, raw float seconds: no clip, no winsorising, no log | `y = max(proxy − delta_hat, 1)`, rounded to int, `proxy = MVT − AOBT_3` | `scripts/stand_ab.py:166-215` (`_own_clock_features`); `scripts/lgbm_submit.py:394-435`; `configs/pipeline_v12.yaml:35-38` |
| matched + ADS-B stage (v11 on) | 161,036 rows moved at 9 airports (v12 adds 5,252 at LIRF) | residual `resid` on top of the matched delta `F` | `rint(max(proxy − (F + resid), 1))`; `resid` is forced to 0 where coverage < 2 or the airport is not gated | `prc/pipeline/adsb_stage.py:1-16, 115-139` |
| unmatched, 9 airports | ≈ 4,900 | schedule-fill mixture `p·sp + (1−p)·nf` | `max(·, 1)` | `scripts/build_submission.py:342-405` |
| unmatched, LIRF | ≈ 380 | same mixture, then four rules in order: fill classifier, date-slip, schedule floor, local-day rule | — | `configs/pipeline_v12.yaml:68-76` |

**Algebra vs ours [X].** Our model A target is `g = LABEL − mvt_minus_aobt3 = AOBT_3 − BLOCK = −delta`
(`src/prc/train.py:1259-1260` at HEAD). It is the same reparameterisation with the sign flipped (slot 21 §6.1).

### 1.2 Model family and key hyperparameters [S]

- **Matched lane:** LightGBM, L2. `learning_rate 0.01`, `num_leaves 255`, `min_data_in_leaf 40`,
  `feature_fraction 0.8`, `bagging_fraction 0.8`/`freq 1`, `cat_smooth 20`, `min_data_per_group 100`,
  `max_cat_threshold 64` (`scripts/lgbm_submit.py:174-177`).
  - Early stopping: cap 50,000 rounds, patience 200 (`:178`), on held-out months 3 and 9.
  - Refit on all 12 months at `best_iter × n_all/n_fit`; v3 used 30,846 trees (slot 21 §4.2).
  - Three seeds (0, 1, 2), averaged on the **delta** scale (`:1026` `mean_delta`).
  - Shipped boosters: `lgbm_v6_queue_order_seed{0,1,2}.txt` (`configs/pipeline_v12.yaml:44-47`). These are
    **unchanged from v6 through v12**.
  - Abort guard: the run stops if the model fails to beat the mean-delta predictor by 15 % on its stopping set
    (`:200`, used `:920`).
- **ADS-B stacker ("ADN"):** a LightGBM residual model `G` plus a torch MLP `N`, blended into `B`.
  - The model files are `G_s{seed}_k{fold}.txt`, `N_s{seed}_k{fold}.pt` and `N_s{seed}_k{fold}_stats.npz`
    (`prc/pipeline/adsb_stage.py:175-180`). The MLP runs in a subprocess (`prc/pipeline/legacy.py:72-78`).
  - Fold gains (`reports/MSE_LEDGER.md:629-634`): G +4,836, N +4,905, B +5,146 weighted fold MSE.
  - **The stacker code itself (`scripts/adsb_stack.py`, the feature builder, `adsb_ship.py`) is not committed.**
    `legacy.py:65-78` imports it read-only from another worker. The blend weights, MLP architecture and feature
    formulas are therefore **[U]**.
- **Unmatched body:** sklearn `HistGradientBoostingRegressor(max_leaf_nodes 15, max_iter 200, min_samples_leaf 100)`.
  - It is fitted on non-fill rows only, with the target winsorised at ±3,000 s **for the fit only**
    (`scripts/build_submission.py:257-296`).
  - The tail uses cell estimates, via `nf_hybrid` (`:307`).
- **Hierarchical cells:** `[ADEP] → [ADEP, sp_band] → [ADEP, sp_fine, dayoff]`, shrunk with `K_SHRINK 5`
  (`:55, 369-381`) on `SP_EDGES` (`:51`).
- **LIRF fill classifier (arm R):** a LightGBM binary model with 15 leaves, lr 0.05, 400 trees
  (`scripts/rome_fill.py:49-51`).
  - Stake weight: `clip(1 + (max(sp,0)/3600)², 1, 25) × 8` on unmatched rows (`:45-46, 66-68`).

### 1.3 Ensembling and blending [S]

- Matched: one pooled LightGBM with 3 seeds. No per-airport models: RESULT 7 found them "NOT WORKING" on LightGBM,
  though they worked on the earlier HGB (slot 21 §3.3). No second learner shipped.
  - CatBoost `C_delta` alone was −1,422 MSE worse.
  - A 0.5·CatBoost + 0.5·CAP blend passed a zero-compute screen at +303.7 weighted fold MSE, below the ship bar
    (`reports/MSE_LEDGER.md:588-593`), and was **not shipped** in the public code.
- ADS-B: the stacker's `G`/`N` blend, bagged over seeds × 5 day-folds.
- Unmatched: a mixture, not a blend.

### 1.4 Post-processing [S]

- A 1-second positivity floor with a tripwire: abort if it binds on ≥ 0.5 % of rows (`scripts/lgbm_submit.py:394-406`).
- **No upper clip, no shrinkage, no per-airport bias shift** on the matched lane.
- LIRF unmatched rules (`configs/pipeline_v12.yaml:72-76`):
  - schedule floor `max(pred, rint(sp))` for 24,000 ≤ sp < 86,400;
  - `rint(sp)` when the schedule falls on the same Europe/Rome local day (RLD, v12);
  - the date-slip two-class expectation.
- ADS-B gate, per airport (`adsb_stage.py:72-92`), set in `MANIFEST.json`:
  - First build: a KS-shift hold rule kept EDDF, LEBL and LSZH on `F`.
  - Then an importance-reweighting check (RWC; `plans/PREREG_reweight_check_2026_09_11.md`) found the 2026 shift
    benign at those three plus EDDM. The gain reweighted to the 2026 feature mix was 1.05–1.25× the unweighted
    gain.
  - v11 final gate: 9 airports, LIRF excluded. v12 adds LIRF after RWC.3 (0.77×).

### 1.5 Validation [S]

- **Fold A:** Jan + Jul 2025 held out; early stopping on months 3 and 9; fit on 8 months (`configs/pipeline_v12.yaml:29-32`).
- LOMO-12 for the unmatched arms.
- Paired bootstrap, date-block for matched and tail arms. Units are MSE, with a close bar of 1,000 MSE and a ship
  bar of +500 (slot 21 §5).
- **ADN folds:** calendar day, `day index mod 5` within each month; 2025-01-09 excluded as the census day
  (`reports/MSE_LEDGER.md:635-638`).
  - The ADN stacker was trained on the **fold-A hold-out rows only** (Jan + Jul 2025, 339,015 matched rows).
  - Its input `F` is arm F's out-of-fold delta (`MSE_LEDGER.md:629-634`; `reports/ADSB_REGIME_CHECK_2026_09_11.md:5`).
  - [X] So the ADS-B model saw two months of labels, not twelve.
- 2025 reproduction guard: the pipeline stage reproduced the stacker's OOF to 1.6e-12 s, RMSE 210.4198
  (`MSE_LEDGER.md:639-648`).

### 1.6 Feature list (shipped arm F = 83 columns + ADS-B stage), mapped to ours

Their lists: `scripts/stand_ab.py:65-81` (BASE, SURF, STANDHIST, ENC), `:92-113` (QUEUE), `:955-967` (ORDER).
Definitions: `:166-215` (own clocks), `:250-330` (surface and stand), `:955-1000` (order). Ours at HEAD
`b5b0be7`:
- `src/prc/features.py:59-103` (clock/static) and `:140-231` (context);
- `src/prc/features_extra.py` (`fl_`, `dup_`, `seq_` groups);
- `src/prc/train.py:382-446` (label encodings);
- production visibility from `configs/main_flseq.yaml` `include_candidates`.

(a) we have an equivalent (ours named), (b) we do not, (c) unclear.

| theirs | meaning | mark | ours |
|---|---|---|---|
| `proxy` | MVT − AOBT_3 | a | `mvt_minus_aobt3` |
| `sp` | MVT − SCHED | a | `mvt_minus_sched` |
| `eobt_p`, `lobt_p`, `iobt_p` | MVT − EOBT_1 / LOBT / IOBT | a | `mvt_minus_eobt1`, `mvt_minus_lobt`, `mvt_minus_iobt` |
| `aobt_sched` | AOBT_3 − SCHED | a | `aobt3_minus_sched` |
| `aobt_eobt` | AOBT_3 − EOBT_1 | a | `aobt3_minus_eobt1` |
| `eobt_lobt` | EOBT_1 − LOBT | a | `lobt_minus_eobt1` (sign flipped) |
| `eobt_sched` | EOBT_1 − SCHED | b in production | `eobt_schedule_revision` exists as a candidate, not in `main_flseq` |
| `lobt_sched`, `iobt_lobt` | LOBT − SCHED, IOBT − LOBT | b | — (theirs: `lobt_sched` single-feature permutation +35.9 s, HGB era, slot 21 §4.1) |
| `airborne3`, `airborne1`, `arvt_diff` | ARVT_3 − MVT, ARVT_1 − MVT, ARVT_3 − ARVT_1 | b | — (`planned_flight_s` = ARVT_1 − EOBT_1 was tested and rejected) |
| `aobt_sec`, `mvt_sec`, `sched_sec` | second-within-minute of each clock | b in production | `proxy_sec_in_min` is a candidate only |
| `hr`, `tmin`, `dow` | hour, minute-of-day, weekday | a | `hour_utc`/`hour_local`, `minute_of_day_utc`, `dow` |
| `doy` | day of year | b | — (`month` is computed but excluded from the model, `train.py:709-719`) |
| `actype_null` | aircraft type missing | b | — (theirs: zero splits) |
| `f_callsign` | CALLSIGN_flt ≠ FLIGHT_mvt | b | — |
| `f_ades` | ADES_FILED_flt ≠ ADES_mvt | c | `diverted` compares ADES_flt with ADES_FILED_flt, a different pair |
| `f_type` | type _flt ≠ _mvt | a | `actype_mismatch` |
| `f_rule` | flight rule _flt ≠ _mvt | b | — |
| `apt_b30`, `arr_b30` | take-offs, arrival in-blocks, previous 30 min | a | `dep_tko_prev_30m`, `arr_inblock_prev_30m` |
| `rwy_b30`, `rwy_rate` | same-runway take-offs, 30 / 60 min | a | `rwy_tko_prev_15m`, `rwy_tko_prev_60m` |
| `n_push` | AOBT_3 pushes in the previous 30 min | b | — |
| `sched_prev60`, `sched_next60`, `sched_day` | scheduled departures ±60 min, per day | b | — |
| `dep_dur` | rolling median proxy over the last 30 departures | a (approx.) | `nbr_mvt_minus_aobt_mean_30m/60m` (time window, mean, lag 60 s) |
| `rwy_dur` | same, per runway | b | — |
| `arr_dur` / `arr_ob60_taxiin` | arrival taxi-in mean, 60 min | a | `arr_taxi_in_mean_60m` |
| `arr_ob60_cdiff`, `arr_ob180_cdiff` | arrivals' in-block − ARVT_3, 60 / 180 min | b | — |
| `prev_arr_gap` | take-off − last arrival in-block at my stand | a | `stand_arrival_gap` |
| `prev_arr_taxiin`, `prev_arr_cdiff`, `prev_dep_gap` | that arrival's taxi-in and clock gap; gap since the previous AOBT_3 at the stand | b | (`ta_` / `occ_` groups tested and rejected) |
| `te_*` × 12 keys | smoothed (k = 50) y encodings: ADEP, STAND, stand prefix, airline, ADES, type, RUNWAY, operator, segment, wake, flight type, `ars` | c | `ref_p10/p20/med/n` (stand×runway), `arh_med` (rwy×hour), `apt_med`; the other keys are native categoricals |
| `de_*` × 12 keys | the same, encoding **delta** | b | — (`op_offset_med` = operator × airport delta median was rejected, −0.17 [−0.40, +0.05]) |
| `q_apt_at_push`, `q_rwy_at_push` | departures pushed but not airborne at my AOBT_3 | b | — |
| `q_pushed_after_me`, `q_rwy_tko_in_taxi`, `q_rwy_push_pre20` | pushes during my taxi; runway take-offs during my taxi; runway pushes 20 min before me | b | — (`queue_ahead` counts SCHED→MVT and was rejected) |
| `q_dep_tko_sym15`, `q_rwy_tko_sym10` | take-offs ±15 min (airport), ±10 min (runway) | a | `dep_tko_prev_15m` + `dep_tko_next_15m`; `seq_*` |
| `q_arr_taxiing_at_push` | arrivals taxiing in at my AOBT_3 | b | — |
| `q_arr_taxiin_sym30_push/_tko` | arrival taxi-in, ±30 min around push / take-off | c | backward only, `arr_taxi_in_mean_30m` |
| `q_rwy_ambient_proxy` | median peer proxy on the runway, ±1 h, self excluded | b | backward airport mean only |
| `q_next_arr_onblock_gap` | next arrival in-block at my stand after AOBT_3 − take-off | b | — |
| `o_dev_mvt`, `o_dev_flt`, `o_n_line` | MVT_ID / FLIGHT_ID deviation from a ±250-row rolling line | b | — (we drop both ids, `features.py:693-700`) |
| **ADS-B, 26 features + coverage code** | known names: `first_ground_rel_aobt3`, `first_move_rel_mvt`, `first_ground_gs`, `dwell_end_rel_aobt3`, `dwell_end_rel_mvt`, `sensor_rel_aobt3`, `path_m`, `taxi_med_gs`, `dwell_to_centroid_m`; coverage 0 = no day file / 1 = not joined / 2 = joined mid-taxi / 3 = seen from the stand | b | our override reads only `t_push` (pushback detection) and a per-airport bias; the full list is [U] |
| unmatched body: `sp`, `dayoff`, `hr`, 5 encodings, `hprox_med`, `hprox_n` | `hprox_med` = airport × UTC-hour median of MVT − AOBT_3 over that hour's matched departures (n ≥ 5), `scripts/unm_congestion.py:13-14, 71-72, 133-160` | a (partial) | model B sees `nbr_mvt_minus_aobt_mean_30m/60m` (backward, lag 60 s); `unmatched_columns` does not drop the `nbr_` prefix (`train.py:470-472`) |

Not shipped in arm F, but built: DAY (9), WEATHER (10), STAND_BLOCK (10). Weather pooled scored +0.49 s
(`README.md:67`; `reports/WHERE_WE_STAND_2026_09_10.md:239`).

Group values from drop-and-refit (`reports/FEATURE_IMPORTANCE_2026_09_10.md:22-23`):
- `clock` +36.6 s [31.5, 42.3];
- `stand_id` +3.9, `stand_hist` +3.3, `airline_seg` +3.3, `queue_counts` +2.6, `ordering` +1.9, `destination` +1.1;
- `actype_wake` +0.95, `recent_taxi` +0.64, `airport` +0.51, `season_time` +0.30, `flags` −0.13.

### 1.7 Every published number [S]

**Board by version** (`reports/MSE_LEDGER.md:30-31, 378-380, 468-469, 517, 526, 709, 740`):

| ver | board | MSE | Δ MSE | change | fold projection → transfer |
|---|---:|---:|---:|---|---|
| v1 | 689.6901 | — | — | constant probe | — |
| v2 | 301.7019 | 91,024 | — | HGB, 26 features + stratum | — |
| v3 | 291.6317 | 85,049 | −5,975 | LightGBM on the E1 cache, matched rows | −5,123 → 1.17× |
| v4 | 289.6732 | 83,911 | −1,138 | stratum hybrid, unmatched rows | — |
| v5 | 288.1406 | 83,025 | −886 | 3 seeds + queue block | −910 → 0.97× |
| v6 | 286.9017 | 82,313 | −712 | arm F (record-ordering block) | −745 → 0.96× |
| v7 | 285.8013 | 81,682 | −630 | Rome fill classifier + date-slip, 382 rows | ≈ −4,150 → 0.15× |
| v9 | 282.6790 | 79,907 | −1,775 | E3C witness on the unmatched body, 4,828 rows | +3,960 registered → 0.45× |
| v10 | 281.5182 | 79,252 | −655 | LIRF schedule floor, 13 rows | partial |
| **v11** | **267.4518** | 71,530.5 | **−7,722.0** | **ADN ADS-B stage, 161,036 matched rows, 9 airports** | **−5,680 → 1.36×** |
| v12 | 266.3738 | 70,955.0 | −575.5 | RLD (3 LIRF rows) + LIRF in the ADS-B gate (5,252 rows) | −664 → 0.87× |

**Ablations and levers with numbers** (fold A, matched lane unless stated):

- `delta` vs `y` target: 285.24 → 252.66 (`README.md:24-25`).
  - At capacity: `y` alone 260.651 vs `delta` 224.258.
  - A 0.5 blend is −9.083 s pooled, but **+9.992 s on |delta| > 20 min** (31.1 % of matched SSE) and −21.08 on
    2–10 min (`plans/PREREG_taxiout_2026_09_08.md:2499-2536`).
- Winsorised target −30.3 s; L1 loss −14.9 s; out-of-fold target encoding −0.96 s; one undivided model −144 s
  (`README.md:67-70`; slot 21 §3.3).
- CAP (127 leaves, 20 min_data, 0.6): +253 weighted fold MSE, not shipped (`MSE_LEDGER.md:568`).
- ADN, re-priced independently: **+5,146.5 [+4,720, +5,618] weighted fold MSE**; matched RMSE 222.563 → 210.494
  (`MSE_LEDGER.md:629-634`).
  - All of the gain sits on covered rows; early-off-block rows account for +1,431.
- **Post-ADN error map** (`reports/POST_ADN_ERROR_MAP_2026_09_11.md` §1, §6):
  - 69 % of the remaining matched error sits on rows ADS-B cannot move.
  - Early rows (δ < −600 s) are 4.4 % of rows and 42 % of error.
  - ADN gain by coverage code: code 3 = 1,887 + 888 + 283; code 2 = 1,355 + 542 + 191.
  - [X] **Code-2 rows (joined mid-taxi, pushback not seen) carry 2,088 of the 5,146 units, 41 %.**
- 2025 → 2026 joined coverage (`POST_ADN_ERROR_MAP §5`): LTFM 0.000 → 0.0001, LFPG 0.05 → 0.12,
  LEMD 0.13 → 0.65, EGLL 0.30 → 0.77, LIRF 0.45 → 0.20. EDDF, LEBL and LSZH "hold".
  - 2025 movable share: EHAM .72, LEBL .79, LSZH .70, EDDM .53, EDDF .51 (`MSE_LEDGER.md:639-648`).
- RWC gains at the disputed airports, reweighted to 2026: 1.05–1.25×. LIRF 0.77×, interval [3,062, 16,565]
  (`MSE_LEDGER.md:683-688`; `plans/PREREG_reweight_check_2026_09_11.md:113-164`).
- Forward fold vs interleaved: 273.65 vs 269.39; "interleaving flatters by +4.26 s"
  (`reports/TRAIN_SERVE_AUDIT.md:53`).

### 1.8 v10 → v11: what changed in code

Commit `f51e303` (2026-09-11 17:39 −0400) carries v11 (`git show --stat`) [S]:

1. **New `prc/pipeline/` package.** It holds a config-driven one-command pipeline (`config.py`, `lanes.py`,
   `models.py`, `ingest.py`, `write.py` and others), with airports and lanes as YAML (`configs/pipeline.yaml`).
   - Its "A1" gate is that the pipeline **reproduces v10 on all 344,841 rows, max |diff| 0 s**
     (`MSE_LEDGER.md:594-596`).
   - The matched lane loads the saved v6 boosters. It does not retrain.
2. **`prc/pipeline/adsb_stage.py` (185 lines), the only scoring change.**
   - Strict readers: the table must hold every scored matched `MVT_ID` exactly once, no label columns, and
     provenance metadata.
   - A gate reader, and a stage that adds the stacker's residual to the matched delta.
   - The residual is forced to 0 on uncovered or ungated rows; changed rows are counted per airport.
   - `lanes.py:444-474` calls it after the matched lane.
3. **Not in the commit:** the ADS-B feature table builder, the stacker (`scripts/adsb_stack.py`), the gate script
   (`adsb_ship.py`) and the check script (`adsb_v11_check.py`).
   - The ledger says the v11 manifest "hashes 17 working-tree files that differ from HEAD (uncommitted code)"
     (`MSE_LEDGER.md:671-677`).
4. **Independent diff of v11 against v10** (`MSE_LEDGER.md:689-700`): 161,036 rows differ, exactly the `+adsb`
   rows.
   - Per airport: EGLL 30,534, EHAM 27,044, LEMD 23,791, LEBL 22,358, EDDM 20,785, LSZH 16,960, EDDF 14,963,
     LFPG 4,596, LTFM 5.
   - Unmatched and LIRF-rule rows are byte-equal to v10.
   - Change: median −17 s, p5 −162 s, p95 +199 s.
5. v11 → v12 (`dc991e0`): `configs/pipeline_v12.yaml`, LIRF added to the gate, the RLD rule, and
   `lanes.py`/`config.py` edits (+24/+25 lines). 5,255 rows differ, all at LIRF (`MSE_LEDGER.md:733-745`).

[X] In summary, the whole 14.07 s step from v10 to v11 is one post-hoc residual model on external data, applied on
top of a frozen matched model. No feature, target or hyperparameter of the base changed.

### 1.9 External data [S]

| source | use | licence as stated |
|---|---|---|
| adsb.lol `globe_history` (2025 and 2026 release tags) | ADS-B stage, v11 on | **ODbL 1.0** (`README.md:74-78`; `scripts/adsb_extract.py:6-8`) |
| Iowa Environmental Mesonet ASOS/METAR | weather block, built but **not shipped** | "public domain — the service states no licence restriction" (`scripts/fetch_weather.py:4-6`) |
| OPDI | feasibility only, not used (`reports/OPDI_FEASIBILITY_2026_09_11.md`) | — |

---

## 2. kind-mango (`2a4ec1b`)

### 2.1 Targets and lanes [S]

| lane | target | rows | where |
|---|---|---|---|
| base outside LIRF, "plain" members | `y`, raw seconds, with the scored flight's NM clocks as features (since v73) | MB3 rows: ADEP ≠ LIRF, 0 < y ≤ 80,000, 1,923,953 served (`models/lgbm_r_all_v85.meta.json` `n_served`) | `RECAP.md:82` (v73); slot 22 §4.2 (MB3) |
| base outside LIRF, "anchored" members | `y − clip(MVT − AOBT_3, 0, 7200)`, rows with AOBT_3 only (1,903,118) | reassembled as `clip(anchor + offset, 0, ∞)`; rows without AOBT_3 keep the plain term | `src_v3/train_v74_anchor.py:1-5`; `src_v3/predict_v67.py:39-46` |
| LIRF | own regime head: `p_fb·sd + (1 − p_fb)·R_norm`, then the Step A band table for null-record rows with sd > 14,400, then the ITY340 rule for sd > 70,000 | — | slot 22 §5; `RECAP.md:92-93` (v63, v64) |

[X] Their anchored target is ours (`g`) with the anchor clipped to [0, 7,200].

### 2.2 Model family and hyperparameters [S]

- **LightGBM, `linear_tree`, "MB5 clock-era" study, v76 onward** (`models/lgbm_r_all_v85.meta.json`):
  - lr 0.01514, 613 leaves, `min_data_in_leaf` 515, `feature_fraction` 0.837, `bagging_fraction` 0.909 / freq 5;
  - `lambda_l1` 0.002, `lambda_l2` 0.027, `min_gain_to_split` 0.348, **`linear_lambda` 15.09**;
  - `cat_smooth` 17.6, `min_data_per_group` 44, `max_cat_threshold` 9;
  - 2,284 early-stopped rounds × 1.478 = 3,377 rounds on the full year, 3 seeds.
- **Tuning:** 24 seeded-TPE Optuna trials (`RECAP.md:79`). Fit on months 2–6 and 8–10, early stop on 11 and 12,
  up to 4,000 rounds. Search bounds: ≤ 1,024 leaves, `min_data_in_leaf` ≤ 2,000, `linear_lambda` ≤ 100.
- **CatBoost:** depth 8, lr 0.08, `l2_leaf_reg` 3, RMSE, od 200 (`src_v3/config.py:120-125`).
  - 5,000-round cap on the hold-out; 7,385 rounds on the full year, scaled by rows (`RECAP.md:88`).
- **Per-airport LightGBM (v89):** one member per airport with the v85 columns and v76 parameters, one seed,
  3,377 / 3,340 rounds (`RECAP.md:68`).
- **Features:** 131 columns (`models/lgbm_r_all_v85.features.txt`; identical in `lgbm_pa_v89.features.txt` and
  `lgbm_r_all_v85a.features.txt`).

### 2.3 Ensembling [S]

Served base outside LIRF, v89 (`src_v3/predict_v67.py:109-138`; `RECAP.md:67-68`):

```
LGB term = 0.5 · plain' + 0.5 · anchored'
           where plain'    = 0.5 · global plain LGB (MS3 over seeds)    + 0.5 · per-airport plain LGB
                 anchored' = 0.5 · global anchored LGB (MS3 over seeds) + 0.5 · per-airport anchored LGB
CAT term = 0.5 · plain CatBoost (clocks) + 0.5 · anchored CatBoost
base     = 0.5 · LGB term + 0.5 · CAT term
```

The weights are the pre-registered 0.5. A grid refit of the three weights scored −2 MSE cross-fitted, and they were
kept (`RECAP.md:78`; `config.py:149-150`). XGBoost as a fifth member got weight 0.05/0, +37 MSE, residual
correlation 0.975 (`RECAP.md:73`).

### 2.4 Post-processing [S] (`src_v3/postprocess.py:1-140`)

- **MS3:** where the seed members disagree by > 3,600 s outside LIRF, serve their median. It moved 4–12 rows per
  version.
- **MS1 per-airport upper bounds:**
  - EDDF, EDDM, LEBL, LEMD: cap at the 2025 tail maximum + 600 s.
  - EGLL, EHAM, LFPG, LSZH, LTFM: cap at the 2025 clean maximum + 600 s, unless the row has a flight record and
    `mvt_eobt1` > 5,400.
- **MS2:** floor at the 2025 clean 0.1 % quantile per airport.
- NM clock features are clipped to ±7,200 s (`config.py:138`).
- No bias shift.

### 2.5 Validation [S]

- Hold-out: Jan + Jul 2025 outside LIRF. Fit on months 2–6 and 8–10 ("v46 months"); early stop on 11 and 12
  (`RECAP.md:67, 88`).
- One member per arm, often at **fixed equal rounds** against a control (for example `RECAP.md:82`, 500 / 1,000 /
  1,500 / 2,000 rounds).
- Bars: 150 MSE on the hold-out before a refit (`config.py:148`); −0.3 s live to accept.
- A label-free check against the previous file before every upload: per-airport mean shift, max |d|, rows above
  7,200 s.
- A six-fold month-pair protocol is defined (`config.py:49-60`) but the served numbers use the single (1, 7)
  hold-out [X].
- Arrival mirror: a taxi-in model trained on 2025 and scored on 2026 arrivals, as a label-bearing drift check
  (slot 22 §6.4).

### 2.6 Feature list (131 columns, `models/lgbm_r_all_v85.features.txt`), mapped to ours

| group (count) | theirs | mark | ours |
|---|---|---|---|
| categoricals (10) | ADEP, ADES, RUNWAY, STAND, AIRCRAFT_TYPE, WK_TBL_CAT, MARKET_SEGMENT, AIRCRAFT_OPERATOR, FLIGHT_TYPE | a | `CATEGORICALS` (`features.py:34-45`) plus `ADES_REGION` |
| | `FLIGHT_RULE_mvt` | b | — |
| time (3) | `hour`, `dow` | a | `hour_utc`/`hour_local`, `dow` |
| | `month` | b | computed, excluded from the model |
| schedule and plan clocks (6) | `sched_delay`, `mvt_eobt1`, `mvt_iobt` | a | `mvt_minus_sched`, `mvt_minus_eobt1`, `mvt_minus_iobt` |
| | `sd_mod_86400` | b | — |
| | `eobt1_sched` | b in production | `eobt_schedule_revision` (candidate only) |
| | `eobt1_iobt` | b | — |
| signed-log copies (5) | `*_slog` | a | `add_signed_log` over `WIDE_RANGE_COLS` (`features.py:234-270, 666`) |
| NM clocks of the scored flight (5, v73) | `mvt_aobt3`, `aobt3_eobt1`, `aobt3_sched`, `mvt_lobt` | a | `mvt_minus_aobt3`, `aobt3_minus_eobt1`, `aobt3_minus_sched`, `mvt_minus_lobt` |
| | `lobt_aobt3` | b | — |
| forward NM windows (8, v85) | `fw_{apt,rwy}{15,30}_{mean,n}`: mean and count of other departures' MVT − AOBT_3 with take-off in (t, t + 15/30 min], per airport and per runway (`src_v3/build_nm_forward_v85.py:1-40`) | **b** | backward only, `nbr_mvt_minus_aobt_mean_{30,60}m`, airport level (`features.py:198-209`) |
| METAR (18) | wind, gust, head/cross wind, visibility, ceiling, precipitation/snow/thunder/freezing, temperature, dew point, `deicing_gate`, `ice_accretion_1hr` (all NaN) | a (mostly) | `use_weather` hourly table (`weather.py:184-262`) |
| | `ceiling_ft`, `low_ceiling` | c | — |
| congestion (13) | `dep_load_prev_15/30/60m`, `arr_load_prev_*`, `dep_same_rwy_prev_*`, `dep_queue_next_10m`, `arr_taxi_in_mean_60m` | a | `dep_tko_prev_*`, `arr_land_prev_*`, `rwy_tko_prev_15m/60m`, `dep_tko_next_10m`, `arr_taxi_in_mean_60m` |
| | `arr_same_rwy_prev_15m`, `arr_same_rwy_next_10m` | c | `seq_arr_*` are runway-arrival sequence columns |
| operator historic encoders (16) | `openc_*`: median, count and std of 2025 taxi per op×apt, op×apt×rwy, op×apt×hour-bin, op, op×apt×stand, plus a global median | b | label encodings only per stand×runway (`ref_*`), rwy×hour (`arh_med`), airport (`apt_med`) |
| OSM geometry (9) | `taxi_dist_m`, `taxi_dist_known` | a | `fe_osm_dist` in the model A we serve (`main_flseq_osm_nolirf`) |
| | `stand_lat/lon`, `rwy_thr_lat/lon`, `taxi_path_length_m`, `n_turns`, `path_vs_haversine` (taxiway-graph path) | b | — |
| advanced (5) | `secs_since_last_dep_same_rwy` | a | `rwy_headway_prev_s` |
| | `rwy_bank_intensity_5m` | a (approx.) | `seq_burst_*` |
| | `sched_pushback_load_15m`, `runway_diversity_prev_30m` | b | (`active_runways_60m` rejected) |
| | `ades_arr_atfm_delay_today` | b | — |
| turnaround (6) | `ground_time`, `slack_eobt`, `slack_sched` | a (partial) | `stand_arrival_gap` |
| | `in_delay`, `taxi_in_prev`, `link_ambiguous` | b | (`ta_` group rejected) |
| disruption (4) | `dep_sd_mean_60m` | c | `nbr_aobt_delay_mean_*` is AOBT − EOBT, not MVT − SCHED |
| | `dep_fnull_60m`, `arr_txi_p90_60m` | b | — |
| | `arr_txi_mean_60m` | a | `arr_taxi_in_mean_60m` |
| tempo, order, quartiles (19, v40/v44) | `nb_eobt_{med,mean,cnt}_{apt30,apt60,rwy30}`, `nb_eobt_p25/p75_*` (neighbour MVT − EOBT_1 statistics) | b | — |
| | `order_later_eobt_30`, `order_total_30`, `stand_gap`, `queue_eobt_mvt` | b | — |
| plan residuals (2) | `plan_taxi_res` (route median taxi residual, clip ±3,600) | b | — |
| | `plan_nm_taxi` = ARVT_1 − EOBT_1 minus its (ADEP, ADES, type) median, clip ±3,600 (`RECAP.md:91`) | b | raw `planned_flight_s` tested and rejected, −0.21 [−0.49, +0.08] (`docs/features_log.md:218-233`) |
| record flags (2) | `flt_null`, `flt_id_null` | a | `aobt3_null`, `nm_matched` |

We have and they lack [S] (ours): `proxy_*` mode flags (candidate only), `fl_*`/`dup_*`/`seq_*`, `stand_arrival_gap`
anchored at take-off, `ref_p10` quantiles, and the `lobt_bound` projection (§4, row 14).

### 2.7 Every published number, v61–v89 (`RECAP.md:66-95`) [S]

| ver | live | Δ | what |
|---|---:|---:|---|
| v61 | 288.32 | +0.99 rej. | v45 parameters on the v57 recipe |
| v63 | 287.09 | −0.24 | Step A reads `R_norm` (18 LIRF rows) |
| v64 | 285.95 | −1.14 | out-of-fold isotonic map on the LIRF gate |
| v65 | 284.74 | −1.22 (−695 MSE) | `plan_nm_taxi` column |
| v66 | 285.07 | +0.33 rej. | METAR at EOBT_1 |
| **v67** | **281.87** | **−2.87 (−1,627 MSE)** | CatBoost 50/50 into the base; hold-out −2,079 at w 0.5, −1,583 at w 0.3, −1,358 alone |
| v68 | 281.90 | +0.03 rej. | CatBoost in LIRF `R_norm` (hold-out −435 did not transfer) |
| v69 / v70 | not built | — | converged CatBoost +92, second CatBoost +71: under the bar |
| v71 | 281.85 | −0.015 (not accepted) | MF2 runway-queue columns; hold-out −596 at 2,000 rounds |
| v72 | not built | — | fallback heads outside LIRF: +292 / +72 MSE, closed |
| **v73** | **270.58** | **−11.29 (−6,237 MSE)** | five NM clocks of the scored flight; hold-out −5,652 at equal rounds, 1.6× transfer |
| **v74** | **266.84** | **−3.73 (−2,005 MSE)** | clocks in CatBoost + anchored LGB member; anchored at w 0.5 hold-out −818 (−641 at 0.3, −384 at 1.0) |
| v75 | 266.15 | −0.70 (−371) | anchored CatBoost member at 0.5 |
| v76 | 265.27 | −0.88 (−465) | Optuna clock-era parameters (`linear_lambda` 15.1); hold-out −965 member, −240 in the blend; transfer 1.41 |
| v77–v84 | not built | — | blend weights −2; constant-leaf member weight 0; `R_norm` tune −251 (held); XGBoost +37; CatBoost `l2` 10 +54; second study −135 in the blend |
| — | — | — | clock-bias maps: +1,531 in-sample, −51 LOMO; neighbour backward clocks +626 (+733 at LFPG); ARVT_3 residual corr 0.005 |
| v83 | 265.41 | +0.14 rej. | arrival-grid link in the LIRF gate (hold-out −277) |
| **v85** | **264.55** | **−0.72 (−381)** | forward NM windows; member −604, blend −208; queue lengths at push/take-off on top −216 vs −208, not kept |
| v87 | 264.86 | +0.31 rej. | forward windows also in CatBoost |
| **v89** | **264.34** | **−0.22 (−115)** | per-airport LGB 50/50; hold-out −657 in mean, alone +1,112; 5 and 60 min forward windows closed |

Other measurements:
- **Clock check on 2025:** MVT − AOBT_3 against the label has MAE 175 s, 21 % of rows within 60 s, raw RMSE
  376 s. On y > 7,200 s rows its median is 1,082 s (`RECAP.md:82`).
- **Error budget of the v76-style blend** on the hold-out outside LIRF: clean 293,832 rows 36,020 MSE (RMSE 205 s),
  fallback 2,559, 35 tail rows 11,962, one 24-h row 20,224. "45 % of the error sits in 36 rows" (`RECAP.md:73`).
- **Earlier board steps** (slot 22 §1.1): v47 12-month refit −2.74 s; v48 MS1 + MS2 −2.33 s; v51 MB3 row filter
  −5.99 s; v26 dropping `ec_*`/`opdi_*` −13.14 s.

### 2.8 External data [S] (`README.md:258-266`)

| source | use | licence as stated |
|---|---|---|
| Iowa State ASOS METAR | served | public domain |
| OurAirports | served (runway thresholds, bearings) | public domain |
| OSM Overpass (parking positions, taxiways, runways) | served | ODbL |
| Eurocontrol NM daily ATFM / pre-departure delay / traffic | LIRF head only (dropped from the base at v26) | "Open (non-commercial-use clause; challenge use confirmed by organiser, 2026-09-17)" |
| OPDI flight events | dropped at v26 | CC-BY 4.0 |
| VRS StandingData | tried, not served | "Open" |
| **ADS-B (adsb.lol)** | **not used**; the team keeps a "stricter exclusion" (`RECAP.md:414-424`) | — |

---

## 3. Other repos, only where neither leader has it

- **joyous-rainbow** (`README.md`, 299.23): no anchored target, and uses `linear_tree` on raw y.
  - They report `linear_tree` alone as a **31.4 s board gain** (333.7 → 301.8).
  - They report capping predictions at the per-airport maximum as **+19.7 s worse**, and dropping the hardest
    airport from training as **+14.6 s worse**.
  - [X] This contradicts kind-mango's MS1 (−2.33 s, bundled) and MB3 (−5.99 s). Both are measured on raw-y models;
    neither applies to our anchored target without a test.
- Nothing else found that neither leader has.

---

## 4. Levers they have that we don't

Board effects are live RMSE or MSE as published. A hold-out number is marked as such.

| # | lever | who | their measured effect | our status |
|---|---|---|---|---|
| 1 | **Learned ADS-B residual stacker** (LightGBM + MLP, 26 features, coverage codes 2 and 3) on top of the matched delta, gated per airport | MQ | **board −7,722 MSE (v10 → v11)**, 1.36× its fold projection; fold +5,146 [+4,720, +5,618]; v12 with LIRF added −575 (bundled) | We serve a **rule override** (`y_a = MVT − t_push + b`, 900 s guard) at 5 airports, pushback-observed rows only, 63,544 rows (`adsb-full-ingest-candidate.md` §0.1). [X] Their code-2 (mid-taxi) rows are 41 % of their fold gain, and our rule cannot act on them. They also move EDDF, LEMD and LFPG, which we do not. |
| 2 | Plain-y member (clock as feature) blended 50/50 with the anchored member | KM | anchored member at w 0.5 hold-out −818 MSE vs plain; v74 −3.73 s live, bundled with clocks in CatBoost. MQ's reverse test: y-blend −9.08 s pooled, +9.99 s on \|delta\| > 20 min | anchored `g` only |
| 3 | CatBoost as a 50/50 blend member (depth 8, lr 0.08, l2 3, ≈ 5–7k rounds) | KM | v67 −2.87 s live (−1,627 MSE), hold-out −2,079; anchored CatBoost v75 −0.70 s | P10 tested CatBoost as a **replacement** (depth 6, lr 0.1/0.2, `max_ctr_complexity` 1) and it lost. **A blend is not recorded** in our results [X] |
| 4 | `linear_tree` with large `linear_lambda` (15) + Optuna retune (613 leaves, lr 0.015) | KM | v76 −0.88 s (−465), hold-out −965; `linear_tree` at v21 −15.6 s (bundled, raw y); joyous-rainbow −31.4 s (raw y) | `use_linear_tree: false` (`train.py:164`); last tried at v4 (09-11, `linear_lambda` 1, pre-flseq); lr 0.05 / 127 leaves / ≤ 2,000 rounds (runs stop at ≈ 900–1,300) |
| 5 | Forward NM-clock windows: other departures' MVT − AOBT_3, take-off in (t, t + 15/30], airport and runway | KM | v85 −0.72 s (−381); hold-out member −604, blend −208 | backward airport windows only |
| 6 | Push-anchored queue block: counts at the row's AOBT_3, pushes during taxi, arrivals taxiing at push | MQ | v5 −886 board MSE, bundled with 3 seeds (fold −910); group drop-refit +2.6 s. KM's AOBT_3/take-off queue lengths on top of forward windows: closed (−216 vs −208) | none (our `queue_ahead` is SCHED→MVT and was rejected) |
| 7 | Record-ordering block (MVT_ID / FLIGHT_ID deviation from the local id line) | MQ | v6 −712 board MSE (fold −745, 0.96×); group +1.9 s | ids dropped from features |
| 8 | Per-airport LightGBM members, 50/50 with the global member | KM | v89 −0.22 s (−115); hold-out −657. MQ: +5.53 s on HGB, "not working" on LightGBM | `airport_models` flag exists (`train.py:258`); no run found in `experiments.jsonl` [X] |
| 9 | `plan_nm_taxi` (filed ARVT_1 − EOBT_1 minus its route × type median) | KM | v65 −1.22 s (−695 MSE), before the clocks | raw `planned_flight_s` rejected; the route-median residual is untested |
| 10 | Delta encodings `de_*` and y encodings `te_*` over 12 keys, k = 50 | MQ | no isolated number (stand-identity group +3.9 s drop-refit) | `ref_*`, `arh_med`, `apt_med`; operator delta median rejected |
| 11 | Operator-keyed historic taxi encoders (median / count / std, 5 keys) | KM | no isolated number (in base since v21); encoder refit variants +1.29 s / +2.25 s worse | not present |
| 12 | Additional clock pairs and flags: `lobt_sched`, `iobt_lobt`, `airborne3/1`, `arvt_diff`, second-in-minute, `f_callsign`, `f_rule`, `eobt1_iobt`, `lobt_aobt3` | MQ, KM | MQ's E1 block (pairwise clocks + arrival/stand) +8.17 s [+6.48, +10.17], HGB era, bundled; permutation `lobt_sched` +35.9 s (HGB) | partly; missing columns listed in §1.6 and §2.6 |
| 13 | Base trained only on served rows (no LIRF, y ≤ 80,000) | KM | v51 −5.99 s (raw-y `linear_tree`, pre-clock) | model A trains on all airports and all labels. [X] The mechanism was linear-leaf rotation, which may not apply to us. |
| 14 | Per-airport caps and floors (MS1/MS2), member-median on disagreement (MS3) | KM | v48 −2.33 s (5 + 104 rows, pre-clock); MS3 moves 4–12 rows | zero floor + LOBT window projection (slack 3,606 s, `train.py:244, 1312-1345`). joyous-rainbow measured a per-airport max cap at +19.7 s worse |
| 15 | LIRF regime head (fill gate × `R_norm` mixture, isotonic OOF map, Step A band table) | KM | v22 −9.8 s; Step A v16 −58 s; v64 −1.14; v63 −0.24 | model B LIRF schedule reparam; `arrival_fill` in progress (another agent) |
| 16 | LIRF stake-weighted fill classifier, date-slip expectation, schedule floor, local-day rule | MQ | v7 −630, v10 −655, v12 −575 (bundled with ADS-B) board MSE | as row 15 |
| 17 | Unmatched body fitted on non-fill rows, winsorised at ±3,000 s for the fit only; hierarchical shrunk cells for the tail | MQ | cells vs plain GBM 1,648.4 vs 1,793.6 (stratum); E3C witness v9 −1,775 board MSE | model B is one LightGBM; it already sees backward `nbr_mvt_minus_aobt_mean` |
| 18 | Taxiway-graph path length, turn count, stand and threshold coordinates (OSM) | KM | no isolated number (in base since v21) | stand-to-runway distance only (`fe_osm_dist`) |

Not levers but methods, with no score attached:
- kind-mango's **arrival mirror** as a 2026 transfer check;
- merry-quicksand's **ADS-B reweighting check** (RWC) to decide per-airport gating from 2025 labels plus the 2026
  feature mix.

---

## 5. What I could not establish

- **ADN internals [U]:** the 26-feature formulas, the MLP architecture, the G/N blend weights and the training
  target's exact form. The code is not committed (`prc/pipeline/legacy.py:65-78`). Only 9 feature names appear in
  committed files.
- **merry-quicksand after v12 [U].** Nothing public covers 266.37 → 250.73.
- **Whether kind-mango's 50/50 plain + anchored blend beats our anchored-only model on our features [U].**
  merry-quicksand's controlled test points the other way on the pooled score. The two results are on different
  feature sets and learners.
- Discord content is still second-hand, relayed through `RECAP.md:381-430`.
