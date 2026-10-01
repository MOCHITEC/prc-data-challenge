# Feature levers L1–L4 from the public-code deep dive: L2 ADOPT in model A; L1, L3, L2-in-model-B REJECT; L4 skipped (2026-09-30)

**Verdicts (prc.compare.verdict, four folds, 672,428 pooled OOF rows):**

| arm | full CI (K=0) | K=200 | month bootstrap | matched lane | verdict |
|---|---|---|---|---|---|
| L1 `plan_nm_taxi` → anchored LGB | −0.03 [−0.11, +0.07] | −0.04 [−0.15, +0.06] | −0.03 [−0.15, +0.09] | −0.04 [−0.15, +0.09] | **REJECT** |
| **L2 forward NM windows → anchored LGB** | **−0.14 [−0.25, −0.05]** | **−0.20 [−0.31, −0.10]** | −0.14 [−0.25, −0.04] | −0.19 [−0.30, −0.06] | **ADOPT** |
| L3 AOBT_3 queue → anchored LGB | +0.04 [−0.07, +0.15] | +0.03 [−0.09, +0.16] | +0.04 [−0.06, +0.13] | +0.05 [−0.09, +0.19] | **REJECT** |
| L2 → model B (`main_flseq`) | +0.35 [−0.22, +1.03] | −0.21 [−0.37, −0.04] | +0.35 [−0.30, +1.04] | +0.00 (identical) | **REJECT** |
| combined | not run: one model-A lever adopted, so the combined arm is the L2 arm (§5) | | | | — |

**For the rebuild:** add `fe_fwd_nm: true` to the anchored LightGBM member (`configs/main_flseq_osm_nolirf_l2.yaml`).
Model B stays as it is. Whether L2 also helps the plain-y LightGBM or the CatBoost member was not measured here.
Nothing was rebuilt, fitted on the full year, or submitted.

**Pre-registration status:** §1–§5 were committed in `81d52da` before any run (code in `2199361`), and have not been
edited since. The results are in §6–§7.

Source of the four levers: `public-code-deep-dive-2026-09-30.md` §1.6, §2.6 and §4 (rows 5, 6, 7, 9), and
`21-repo-merry-quicksand.md` §4.1 for the written definitions of the queue and ordering blocks. **No competitor code
was opened or copied.** Everything here is implemented from those written definitions in our own code
(`src/prc/levers.py`).

## 1. Existence checks

Searched `src/prc/features.py`, `src/prc/features_extra.py`, `src/prc/train.py`, `docs/features_log.md` and the
deep-dive note's feature table.

| lever | what we have | verdict |
|---|---|---|
| **L1** `plan_nm_taxi` | `features.py` computes the raw plan as a candidate: `(pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("planned_flight_s")`. `features_log.md` 実験 10: "`planned_flight_s` … 200（判定レベル） −0.21 [−0.49, +0.08] … 棄却". Deep-dive table: "`plan_nm_taxi` … b … raw `planned_flight_s` tested and rejected"; §4 row 9: "the route-median residual is untested". No route-median residual exists anywhere in `src/`. | **absent → test** |
| **L2** forward NM windows | Only backward: `features.py` `_window_mean(t, dep_times, mvt_minus_aobt, w * 60, lag_sec=60)` → `nbr_mvt_minus_aobt_mean_{30,60}m`, "Mean and count of `values` whose time is in [t-w-lag, t-lag)", airport level. Deep-dive: "forward NM windows (8, v85) … **b** … backward only". `dep_tko_next_{w}m` counts future take-offs but carries no NM clock. | **absent → test** |
| **L3** AOBT_3-anchored queue | `features_extra.py` `_backlog`: `out["bk_arr_taxiing_at_aobt"] = np.where(np.isfinite(ao), insys(ao), np.nan)` (arrivals taxiing at my AOBT_3) — a hidden candidate; the `bk` group was rejected ("Single groups vs base: … bk +0.16 REJECT", `configs/main_flseq.yaml`). `queue_ahead` counts SCHED→MVT and was rejected ("`queue_ahead` … −0.16 [−0.40, +0.08] … 棄却"). Take-off-anchored counts exist (`dep_tko_prev/next_*`, `rwy_tko_prev_15m/60m`). Deep-dive: "`q_apt_at_push`, `q_rwy_at_push` … b"; "`q_pushed_after_me`, `q_rwy_tko_in_taxi`, `q_rwy_push_pre20` … b"; §4 row 6 "none". | **absent (one of six columns exists as a rejected candidate) → test** |
| **L4** record ordering | `features.py` `numeric_feature_columns`: `exclude = set(BLANKED_DEP_COLS) \| {"MVT_ID_mvt", "FLIGHT_ID_mvt", ...}`. Deep-dive: "`o_dev_mvt`, `o_dev_flt`, `o_n_line` … b … we drop both ids". | absent, but see §2 |

## 2. L4 validity check: is the id ordering scheme the same on the scored set? **No → L4 skipped**

Measured on the raw files (`data/raw/training_*.parquet`, 4,167,797 rows; `data/raw/ranking.parquet`, 689,534 rows).
Scratch scripts `idcheck*.py` in the session scratchpad; the numbers below are their output.

1. **Structure that matches.** Rows adjacent in MVT_ID (gap ≤ 2) share airport and phase in 100.00 % of pairs, in
   both sets. MVT_ID runs are per airport × phase and roughly a month long in both (the largest run is LTFM ARR,
   25,008 rows over 32 days in 2025-08 and 25,166 rows over 32 days in 2026-07). FLIGHT_ID is LOBT-ordered in both:
   median airport-day Spearman(FLIGHT_ID, LOBT) 0.99997 train, 0.99997 rank.
2. **The within-run ordering key differs by airport, and at LTFM it changes in the scored months.** Share of
   id-adjacent departure pairs (same airport) whose SCHED_TIME is non-decreasing, by month:

   | month | EDDF | EGLL | LFPG | LSZH | EHAM | **LTFM** |
   |---|---:|---:|---:|---:|---:|---:|
   | 2025-01 … 2025-08 (range) | 0.995–0.997 | 0.996–0.998 | 0.870–0.883 | 0.539–0.560 | 0.625–0.649 | **0.506–0.521** |
   | 2025-09 … 2025-12 (range) | 0.996–0.997 | 0.998 | 0.872–0.878 | 0.539–0.560 | 0.509–0.708 | 0.618–0.632 |
   | **2026-01 (scored)** | 0.996 | 0.998 | 0.874 | 0.564 | 0.606 | **0.998** |
   | **2026-07 (scored)** | 0.996 | 0.997 | 0.883 | 0.560 | 0.539 | 0.635 |

   EDDM, LEBL, LEMD and LIRF behave like EDDF (0.997–0.998 in train Jan+Jul and in rank). At LTFM the ids are
   essentially unordered by schedule in every training month (0.51–0.63) and **schedule-sorted in 2026-01 (0.998)**,
   a scheme no training month shows. LTFM ARR tells the same story on a label-bearing row type: MVT_ID-adjacent
   arrival pairs non-decreasing in landing time 0.523 (train Jan+Jul) vs 0.791 (rank).
3. **Label-bearing consequence (arrivals, whose taxi-in is visible in both sets).** The ±250-row rolling-line
   deviation of MVT_ID (§3 L4 definition, rows sorted by movement time within airport × phase) against arrival
   taxi-in, Spearman: LTFM **−0.029 (train Jan+Jul) / +0.032 (train all) vs +0.276 (rank)**. At the other nine
   airports the train and rank values agree within 0.04 (e.g. LSZH +0.105 / +0.119, LFPG +0.085 / +0.068).
4. Id gaps are also denser on the scored set: runs of id gap ≤ 2 have median 58 rows in rank against 254 in
   train Jan+Jul (1,364 vs 742 runs for the same row count).

The ordering scheme is not the same on the scored set (LTFM, the airport with the most departures, changes from
unordered to schedule-sorted in 2026-01, and the feature's relation to a label changes sign and size there).
**Per the instruction, L4 is skipped.** No L4 code was written.

## 3. Definitions (as implemented in `src/prc/levers.py`)

Notation per departure i: `t_i` = MVT_TIME (take-off), `a_i` = AOBT_3, same departure airport (ADEP) unless
stated, j ≠ i. Every column is built per frame (training frame from training rows, ranking frame from ranking rows),
like `prc.features.context_features`. Only columns present in ranking.parquet are read; no departure BLOCK_TIME or
TAXITIME is read (tested).

**L1 `plan_nm_taxi`** (1 column). `p_i = ARVT_1 − EOBT_1` in seconds. `med` = median of `p` over
(ADEP, ADES, AIRCRAFT_TYPE_mvt), fitted once on the union of training and ranking departures (label-free; the same
rule as `proxy_mode_table`), groups with ≥ 5 rows. Fallback (ADEP, ADES) median with ≥ 5 rows, else null.
`plan_nm_taxi = clip(p − med, −3600, 3600)`. Null on every unmatched row (EOBT_1 and ARVT_1 are null on all 22,470
training and 5,290 scored unmatched departures).

**L2 forward NM windows** (8 columns). For W ∈ {15, 30} min and group g ∈ {airport, airport × runway}:
`fw_{g}{W}_mean` = mean over departures j in group g with `t_j ∈ (t_i, t_i + W]` and non-null AOBT_3 of
`clip(t_j − a_j, −7200, 7200)`; `fw_{g}{W}_n` = the number of such j. Mean is null when n = 0. Runway columns are null
when RUNWAY is null. Available on unmatched rows too (the peers' clocks, not the row's own).

**L3 AOBT_3-anchored queue** (6 columns; null when `a_i` is null, so null on every unmatched row):

| column | definition |
|---|---|
| `qp_apt_at_push` | #{j: a_j < a_i < t_j} (j with AOBT_3) |
| `qp_rwy_at_push` | the same, j on my runway |
| `qp_pushed_after_me` | #{j: a_i < a_j < t_i} |
| `qp_rwy_tko_in_taxi` | #{j on my runway: a_i < t_j < t_i} (every departure, with or without AOBT_3) |
| `qp_rwy_push_pre20` | #{j on my runway: a_i − 1200 ≤ a_j < a_i} |
| `qp_arr_taxiing_at_push` | #{arrivals into ADEP: landing ≤ a_i < in-block, 0 < in-block − landing < 7200} |

The last one is the same quantity as the existing hidden candidate `bk_arr_taxiing_at_aobt`; it is rebuilt inside
the lever so the L3 arm is one flag. Not taken from the source list: `q_dep_tko_sym15` and `q_rwy_tko_sym10` (we
have the take-off-anchored equivalents), `q_arr_taxiin_sym30_*` (means, not counts), `q_rwy_ambient_proxy`
(its ±1 h window overlaps L2) and `q_next_arr_onblock_gap`.

Real-data sanity (all three levers built on `data/work_flseq`, 9 s): null rates on matched rows train / rank —
`plan_nm_taxi` 0.18 % / 0.17 %, `fw_apt15_mean` 0.88 % / 1.00 %, `fw_rwy15_mean` 3.25 % / 3.50 %, all `qp_*` 0 / 0.
Spearman with model A's target g on training matched rows: `plan_nm_taxi` −0.066 (raw `planned_flight_s` +0.054),
`fw_apt30_mean` −0.192, `qp_pushed_after_me` −0.206, `qp_apt_at_push` −0.131. These are not effect estimates.

## 4. Lanes and flags

| lever | flag | model A (matched) | model B (unmatched) |
|---|---|---|---|
| L1 | `fe_plan_nm_taxi` | yes | no (null there) |
| L2 | `fe_fwd_nm` / `fe_fwd_nm_unmatched` | `fe_fwd_nm` | `fe_fwd_nm_unmatched` — **L2 enters model B**, as kind-mango's forward windows sit in a base that serves rows without AOBT_3 |
| L3 | `fe_push_queue` | yes | no (null there) |

All flags default to off and are in `_HASH_OPTIONAL`. With them off, the columns are never built
(`prc.pipeline.main`) and are hidden if present (`visible_numeric_columns`). `tests/test_levers.py`:
- every one of the 116 recipes in `configs/` keeps its experiment id computed at `d8775f9`;
- `predict_fold` on a frame carrying all 15 lever columns, flags off, gives predictions and `info` equal to the
  frame without them (byte identity);
- with a model-A lever on, model B's predictions equal the flag-off run; with `fe_fwd_nm_unmatched` alone, model A's
  predictions equal the flag-off run;
- hand-built values for every column, an exact brute-force check of the in-system count, and no departure BLOCK read.

## 5. Arms and decision rule

Four folds (jan, jul, jul_fwd, forward), 672,428 pooled OOF rows, `--work-dir data/work_flseq`, `--skip-predict`,
one run at a time. Bases are the archived OOF files (the flags-off byte identity is tested, and every lever arm's
untouched lane must reproduce its base to the last digit, which is checked and reported).

| arm | config | base | question |
|---|---|---|---|
| L1 | `main_flseq_osm_nolirf_l1` | `main_flseq_osm_nolirf-926741d779` | L1 in the anchored LGB member |
| L2 | `main_flseq_osm_nolirf_l2` | same | L2 in the anchored LGB member |
| L3 | `main_flseq_osm_nolirf_l3` | same | L3 in the anchored LGB member |
| L2-B | `main_flseq_l2b` | `main_flseq-8615e6a194` | L2 in model B (the served unmatched lane); model A byte-identical, so every differing row is unmatched |
| combined | `main_flseq_osm_nolirf_` + adopted | `main_flseq_osm_nolirf-926741d779` | final confirmation of the adopted model-A levers together |

**Decision rule, applied per lever:** `prc.compare.verdict` on all rows of the arm against its base — the full-data
CI (K = 0) and the drop-200 CI (K = 200) must both exclude zero on the same side, and that side must be improvement.
For the three model-A arms the matched-lane CI is reported beside it (the unmatched rows are identical by
construction, so all rows = matched rows plus zeros). For L2-B the verdict on `main_flseq → main_flseq_l2b` decides
whether L2 enters model B. Also reported per arm: the month bootstrap, the lane split, per-calendar-month and
per-airport deltas (`scripts/plainy_blend.py detail`).

**Combined arm.** Run only if two or more model-A levers are adopted (with one, it is that arm; with none, there is
nothing to combine). It must itself pass the same verdict against the base. If it does not, the result is reported
and the recommended set for the rebuild is the single adopted lever with the most negative full-data difference.

**Out of scope here:** rebuilding the three-member blend, the plain-y member and CatBoost, any full fit, any
submission or upload. The note reports which levers were adopted; the rebuild is scheduled separately. No
noise-control arm is run (the arms follow the requested base → +lever design; the ±0.5 s per-column noise floor in
`features_log.md` was measured at one seed on the forward fold only, and these arms are three seeds on four folds).

Published effects, for scale only (other teams, other bases): L1 kind-mango v65 −1.22 s (before their clocks); L2
kind-mango v85 −0.72 s (member −604 MSE on their hold-out); L3 merry-quicksand v5 −886 MSE, bundled with 3 seeds.

## 6. Run log

`python -m prc.pipeline --config configs/<arm>.yaml --work-dir data/work_flseq --skip-predict`. The runs were
sequential and nothing else trained alongside them. The feature cache was reused (its fingerprint is unchanged).
Peak cgroup `total_rss` was sampled every 5 s from the `memory.stat` of the parent of the cgroup that
`/proc/self/cgroup` names. The limit is 13,681 MB.

| arm | exp id | CV seconds | peak total_rss |
|---|---|---:|---:|
| L1 | `main_flseq_osm_nolirf_l1-50ebc5643d` | 4,081 | 10,783 MB |
| L2 | `main_flseq_osm_nolirf_l2-ef204b44ff` | 4,134 | 11,231 MB |
| L3 | `main_flseq_osm_nolirf_l3-d08ab29d15` | 3,821 | 11,287 MB |
| L2-B | `main_flseq_l2b-e40f03b86d` | 3,826 | 11,207 MB |

**Reproduction check.** In each model-A arm, the untouched lane reproduces its base to the last digit:

- **L1, L2, L3.** Unmatched RMSE equals the base in every fold: jan 2873.79, jul 1227.62, jul_fwd 1282.18,
  forward 980.73. The unmatched difference is +0.00 [+0.00, +0.00].
- **L2-B.** Matched RMSE equals `main_flseq` in every fold: jan 201.67, jul 241.90, jul_fwd 250.57, forward 197.14.
  The matched difference is +0.00.

The archived base OOF files are therefore valid comparators. Breakdowns come from
`python -m prc.compare` and `scripts/plainy_blend.py detail --scratch experiments`. Logs are in `logs/levers/`
(git-ignored).

## 7. Results

### 7.1 Per-fold RMSE (all rows / matched)

| fold | base `main_flseq_osm_nolirf` | L1 | L2 | L3 |
|---|---|---|---|---|
| jan | 342.54 / 201.52 | 342.35 / 201.20 | 342.54 / 201.51 | 342.58 / 201.58 |
| jul | 297.23 / 241.73 | 297.28 / 241.79 | **297.01 / 241.45** | 297.40 / 241.95 |
| jul_fwd | 308.66 / 250.24 | 308.64 / 250.22 | 308.63 / 250.19 | 308.57 / 250.12 |
| forward | 217.35 / 196.93 | 217.38 / 196.97 | **217.16 / 196.73** | 217.28 / 196.86 |
| overall | 273.72 | 273.69 | **273.57** | 273.75 |

L2-B against `main_flseq`, all rows / unmatched:

| fold | `main_flseq` | L2-B |
|---|---|---|
| jan | 342.16 / 2867.76 | 342.45 / 2871.52 |
| jul | 295.32 / 1203.41 | 295.70 / 1207.95 |
| jul_fwd | 308.29 / 1274.69 | 309.91 / 1293.67 |
| forward | 217.20 / 972.67 | 217.59 / 982.20 |
| overall | 272.96 | 273.31 |

### 7.2 L1 `plan_nm_taxi`: REJECT

- **Curve.** K=0 −0.03 [−0.11, +0.07], P(B better) 0.742. K=20 −0.04. K=200 −0.04 [−0.15, +0.06]. K=2000 −0.02.
- **Month bootstrap.** −0.03 [−0.15, +0.09].
- **Lanes.** Matched −0.04 [−0.15, +0.09]. Unmatched +0.00 (identical). All rows except LIRF −0.09 [−0.18, −0.02].
- **Per calendar month** (all rows / matched):
  - 2025-01: −0.19 [−0.41, −0.06] / −0.33 [−0.58, −0.10]
  - 2025-07: +0.04 [−0.16, +0.26] / +0.06
  - 2025-11: −0.09 [−0.25, +0.07] / −0.10
  - 2025-12: +0.15 [−0.08, +0.36] / +0.16
- **Per airport** (all rows, ordered by share of squared disagreement):
  - LTFM +0.19 [−0.09, +0.41]
  - LIRF +0.29 [−0.12, +0.74]
  - LFPG −0.02
  - EGLL −0.11
  - **EHAM −0.91 [−1.19, −0.60]**
  - EDDM −0.33 [−0.74, +0.04]
  - EDDF −0.12
  - LEMD +0.20 [−0.02, +0.38]
  - LEBL +0.17
  - LSZH −0.18
- **Reading.** The month signs are mixed. One airport (EHAM) improves clearly and is offset elsewhere. Kind-mango's
  −1.22 s came before they had the NM clocks as features. Our model already carries those clocks.

### 7.3 L2 forward NM windows in model A: ADOPT

- **Curve.** K=0 **−0.14 [−0.25, −0.05]**, P(B better) 1.000. K=20 −0.17 [−0.28, −0.07]. K=200
  **−0.20 [−0.31, −0.10]**. K=2000 −0.26 [−0.36, −0.17].
- **Month bootstrap.** −0.14 [−0.25, −0.04].
- **Verdict.** "adopt (−0.14 s, still −0.20 s at 200 rows dropped)".
- **Lanes.** Matched **−0.19 [−0.30, −0.06]**. Unmatched +0.00 (identical). All rows except LIRF −0.21 [−0.31, −0.11].
- **Per calendar month** (all rows / matched). The point estimate improves or stays flat in all four months:
  - 2025-01: −0.01 [−0.16, +0.13] / −0.01
  - 2025-07: −0.23 [−0.46, −0.02] / −0.28 [−0.54, −0.00]
  - 2025-11: −0.29 [−0.46, −0.09] / −0.33 [−0.53, −0.10]
  - 2025-12: −0.08 [−0.30, +0.15] / −0.09
- **Per airport** (all rows, ordered by share of squared disagreement). Seven of ten improve. LTFM, LIRF and LEMD
  read worse, and no CI there excludes zero:
  - **EGLL −0.82 [−1.13, −0.50]**
  - **LFPG −0.22 [−0.59, −0.02]**
  - LTFM +0.16 [−0.15, +0.45]
  - LIRF +0.18 [−0.19, +0.60]
  - EHAM −0.27 [−0.50, +0.01]
  - EDDM −0.18
  - EDDF −0.07
  - LEMD +0.05
  - LEBL −0.26 [−0.53, +0.03]
  - **LSZH −0.36 [−0.68, −0.11]**
- **Size.** Kind-mango published −0.72 s on the board for v85 (−604 MSE for the member on their hold-out). We measure
  −0.14 s on the member. This is the member alone, not the blend.

### 7.4 L3 AOBT_3-anchored queue: REJECT

- **Curve.** K=0 +0.04 [−0.07, +0.15], P(B better) 0.236. K=200 +0.03 [−0.09, +0.16]. K=2000 −0.07 [−0.16, +0.03].
- **Month bootstrap.** +0.04 [−0.06, +0.13].
- **Lanes.** Matched +0.05 [−0.09, +0.19]. Unmatched +0.00 (identical). All rows except LIRF −0.02 [−0.13, +0.08].
- **Per calendar month** (all rows): 2025-01 +0.03, 2025-07 +0.17 [−0.08, +0.41], 2025-11 −0.07, 2025-12 −0.05.
- **Per airport.** EGLL improves, −0.65 [−1.03, −0.19], as does LFPG, −0.18 [−0.51, −0.00]. Seven airports read
  worse:
  - LIRF +0.35 [−0.03, +0.74]
  - EDDF +0.26 [−0.01, +0.57]
  - EHAM +0.26
  - LSZH +0.24 [−0.02, +0.49]
  - LEBL +0.17
  - LEMD +0.16
  - EDDM +0.16
  - LTFM +0.09
- **Reading.** Merry-quicksand's −886 MSE was bundled with three seeds. Our baseline already has three seeds, plus
  the take-off-anchored counts and the `seq_` group.

### 7.5 L2 in model B: REJECT

The comparison is `main_flseq` → `main_flseq_l2b`. Model A is byte-identical, so every differing row is unmatched.

- **Curve.** K=0 +0.35 [−0.22, +1.03], P(B better) 0.176. K=20 −0.20 [−0.49, +0.11]. K=200 −0.21 [−0.37, −0.04].
  K=2000 −0.14 [−0.21, −0.07].
- **Month bootstrap.** +0.35 [−0.30, +1.04].
- **Verdict.** "no difference on all rows". The K=200 CI excludes zero on the improving side, but the full CI does
  not, and the full-data point estimate is worse.
- **Lanes.** Unmatched +4.97 [−4.16, +16.74]. Matched +0.00. All rows except LIRF −0.25 [−0.47, −0.06].
- **Unmatched rows per airport.** LIRF carries 46.1 % of the disagreement and reads +51.03 [−0.69, +108.02].
  Six airports improve with CIs that exclude zero:
  - LTFM −19.92
  - EHAM −12.83
  - EDDM −16.22
  - LSZH −21.05
  - EDDF −6.64
  - LEMD −17.35

  LEBL is −15.46 with a CI crossing zero, and LFPG and EGLL are +0.41 and +5.05.
- **Per calendar month** (all rows): 2025-01 +0.29, 2025-07 +0.38, 2025-11 +1.43 [−1.01, +4.80],
  2025-12 −0.67 [−1.86, +0.65].
- **Reading.** This is the LIRF-unmatched pattern that `osm-on-flseq.md` also found. A LIRF-excluded variant would
  be post hoc and was not run.

### 7.6 Combined arm

Not run. Per §5, the combined arm only runs when two or more model-A levers are adopted. Only L2 was adopted, so
the combined arm is the L2 arm, `main_flseq_osm_nolirf_l2-ef204b44ff`.

### 7.7 What is adopted

| lever | lane | adopted |
|---|---|---|
| L1 | model A | no |
| **L2** | **model A** (anchored LightGBM member) | **yes** |
| L2 | model B | no |
| L3 | model A | no |
| L4 | — | skipped (ordering scheme differs on the scored set, §2) |
