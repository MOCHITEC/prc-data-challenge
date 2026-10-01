# Feature levers L1–L4 from the public-code deep dive: pre-registration (2026-09-30)

**Status: PRE-REGISTERED. Nothing below §5 has been run.** This file is committed before any training run. The
results are appended in §7 afterwards; §1–§6 are not edited after the commit except to fix typos.

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

(filled in after the runs)

## 7. Results

(filled in after the runs)
