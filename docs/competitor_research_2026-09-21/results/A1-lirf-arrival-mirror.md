# A1, the LIRF arrival-mirror fill, on v16 and on the OSM model A: REJECT on both; hybrid_A1 = H2 (2026-09-30)

**Verdicts (prc.compare.verdict, four folds, 672,428 pooled OOF rows):**

| comparison | full CI (K=0) | K=200 | month bootstrap | verdict |
|---|---|---|---|---|
| `main_flseq` → `main_flseq_a1` | −0.09 [−0.21, +0.03] | −0.09 [−0.21, +0.04] | −0.09 [−0.20, +0.01] | **REJECT** (no difference on all rows) |
| `main_flseq_osm_nolirf` → `main_flseq_osm_nolirf_a1` | −0.04 [−0.14, +0.06] | −0.07 [−0.18, +0.05] | −0.04 [−0.13, +0.07] | **REJECT** (no difference on all rows) |
| `hybrid_H2` → `hybrid_A1` (by the rule in §4) | +0.00 [+0.00, +0.00] | +0.00 | +0.00 | **REJECT**. The two files are identical. |

The sign is on the improving side, and the size is in the range the note pre-registered. The note gave lane A
−0.10 to −0.30 s with a point of −0.15. We measured lane A at −0.12 on v16 and −0.05 on the OSM model. Neither CI
excludes zero. The note's own kill condition (a) also fails on both configs: LIRF lane A gets worse in the jul fold.
Following the pre-stated rule, both lanes of the hybrid keep their non-A1 source. `outputs/hybrid_A1.parquet`
therefore has the same content as `outputs/hybrid_H2.parquet`. Nothing was submitted.

## 1. What A1 is (the source)

`docs/competitor_research_2026-09-21/results/lirf-lfpg-mechanisms.md` §5, "A1 — arrival-mirror fill state (the only
candidate)", with the columns defined in §2.1:

- **Mechanism.** At LIRF the recording process writes the schedule as the block time. When it does that for a
  flight's inbound arrival, the departure of the same rotation is 2.6× as likely to be filled (0.399 vs 0.156). The
  arrival record is not blanked.
- **Columns.** `arr_link_fill`, `arr_link_bs_abs`, `arr_link_same_op` and `arr_opday_fill`. They are built from the
  ARR rows of the same frame: train from training arrivals, rank from ranking arrivals. They join on the same-stand
  link: the latest arrival in-block before take-off, within 6 h. The builder is `build_arrival_fill` in
  `scripts/lirf_lfpg_mechanisms.py`. A fill is `|in-block − sched| ≤ 12 s`. `arr_opday_fill` is the operator ×
  airport × UTC-date arrival fill rate, shrunk toward the airport-day rate with k = 5.
- **Model columns, not a post-process** (§2.4). **Lane A only.** The note lists "Arrival columns for the unmatched
  lane" under *Not recommended* (§2.5, +0.0009 AUC).
- **Pre-registered effect.** Lane-A OOF −0.10 to −0.30 s (point −0.15). Board −0.05 to −0.17 s (point −0.08).
- **Kill conditions.** (a) LIRF lane-A RMSE must fall in each of jan, jul and forward, and in jul_fwd. (b) The pooled
  change must be negative on the paired bootstrap. (c) An arity-matched permuted twin must not read within 50 % of
  the arm. "If jul_fwd reads ≥ 0, close it."
- **Optional variant** (same columns nulled outside LIRF). Not run.

## 2. Implementation

- `src/prc/arrival_fill.py` holds `build_arrival_fill`, moved from the script. The only changes: join keys are cast
  to strings, and only the four named columns are attached. On the real frames its output is **equal to the script's**
  (`DataFrame.equals`), on train and on rank. LIRF lane-A link coverage is 0.8793 on train and 0.8777 on rank, which
  reproduces the note's 0.879 / 0.878. The module deliberately lives outside `features.py` / `features_extra.py`,
  because those files are hashed into the feature-cache fingerprint. Existing caches stay valid.
- `TrainConfig.fe_arr_fill` (default False) and `fe_arr_fill_permute` (default False; for kill condition (c), which
  requires `fe_arr_fill`). Both are in `_HASH_OPTIONAL`, so existing experiment ids are unchanged.
  `prc.pipeline.main` attaches the columns before the raw frames are freed, and only when the flag is on. Each frame
  is linked to its own arrivals.
- With the flag off, `visible_numeric_columns` hides the four columns even if they are present. With it on,
  `unmatched_columns` still excludes them, so model B is untouched. `predict_fold` refuses a flag-on run whose frames
  lack the columns.
- Configs: `configs/main_flseq_a1.yaml` and `configs/main_flseq_osm_nolirf_a1.yaml`. Each is its base config plus
  `fe_arr_fill: true`, with a new name.
- Tests: `tests/test_arrival_fill.py`, 9 tests.
  - Link, window and shrinkage on a hand-built frame.
  - Departure BLOCK is never read.
  - Permutation keeps each month's values.
  - The ids of `main_flseq` (8615e6a194), `main_flseq_osm` (bb204cd922) and `main_flseq_osm_nolirf` (926741d779)
    are unchanged.
  - The A1 configs differ from their bases only by the key.
  - **Byte-identity with the flag off:** `predict_fold` on a frame carrying the four columns gives predictions and an
    `info` record equal to the run without them.
  - With the flag on, model A gets the columns, and model B's predictions equal the flag-off run.
- Model B was also unchanged in both real runs. In every fold the unmatched RMSE equals its base to the last digit.
  All 5,290 unmatched rows of each full-fit file equal their base file. The ADS-B override was not applied to any
  file here.

Runs: `--work-dir data/work_flseq`, the same cache as the base runs. Ids: `main_flseq_a1-c14b85d14f` and
`main_flseq_osm_nolirf_a1-e0daec598e`. The first attempt at the OSM run was OOM-killed (exit 137) in its first fold,
because this session ran the test suite beside it. The rerun was clean (exit 0). Other agents' ADS-B scripts shared
the CPU, so the wall clock was about 2× the base runs (7,750 s and 8,254 s of CV). The peak cgroup `total_rss` seen
was 11.78 GB, against a limit of 13.68 GB.

## 3. Results

### 3.1 `main_flseq` → `main_flseq_a1` (v16 + A1)

| fold | A | B | matched A → B | unmatched |
|---|---:|---:|---|---:|
| jan | 342.16 | 342.05 | 201.67 → 201.48 | 2867.76 (identical) |
| jul | 295.32 | 295.37 | 241.90 → 241.97 | 1203.41 (identical) |
| jul_fwd | 308.29 | 308.21 | 250.57 → 250.48 | 1274.69 (identical) |
| forward | 217.20 | 217.00 | 197.14 → 196.92 | 972.67 (identical) |
| **overall** | **272.96** | **272.87** | | |

`prc.compare`: K=0 −0.09 [−0.21, +0.03], P(B better) 0.950. K=20 −0.11 [−0.24, +0.01]. K=200 −0.09 [−0.21, +0.04].
K=2000 −0.15 [−0.24, −0.07]. Month bootstrap −0.09 [−0.20, +0.01]. **VERDICT: REJECT**, "no difference on all rows".

- **Lane split.** Matched −0.12 [−0.27, +0.02]. Unmatched +0.00 (identical by construction). All rows except LIRF
  −0.02 [−0.13, +0.10].
- **Per calendar month** (all rows / matched):
  - 2025-01: −0.11 [−0.30, +0.05] / −0.19 [−0.44, +0.08]
  - 2025-07: **+0.05** [−0.21, +0.30] / +0.07 [−0.26, +0.36]
  - 2025-11: −0.20 [−0.42, +0.00] / −0.23 [−0.46, +0.02]
  - 2025-12: −0.19 [−0.45, +0.05] / −0.21 [−0.48, +0.05]
- **Per airport** (all rows, matched lane in brackets where it differs; the unmatched lane is 0 everywhere; ordered
  by share of squared disagreement):
  - LIRF −0.52 [−1.07, −0.03] (matched −0.80 [−1.51, +0.10]; 20.5 % of the disagreement)
  - LTFM −0.24 [−0.52, +0.11]
  - LFPG +0.01 [−0.24, +0.25]
  - EGLL −0.28 [−0.66, +0.07]
  - EHAM **−0.34** [−0.59, −0.12]
  - EDDF **+0.50** [+0.17, +0.91]
  - LEMD +0.21 [+0.00, +0.41]
  - EDDM +0.06 [−0.38, +0.45]
  - LEBL **+0.26** [+0.04, +0.49]
  - LSZH −0.14 [−0.51, +0.17]
- **Kill condition (a), LIRF lane-A RMSE:**
  - jan 283.22 → 281.98 (−1.24)
  - **jul 439.15 → 439.58 (+0.43)**
  - forward 299.95 → 298.17 (−1.78)
  - jul_fwd −0.24. This is read from the fold record's LIRF all-rows RMSE, 725.07 → 724.83. The unmatched rows are
    identical, so the sign is lane A's.
  - (a) **fails** on jul. (b) also fails: the pooled CI crosses zero.

### 3.2 `main_flseq_osm_nolirf` → `main_flseq_osm_nolirf_a1` (OSM model A + A1)

| fold | A | B | matched A → B | unmatched |
|---|---:|---:|---|---:|
| jan | 342.54 | 342.45 | 201.52 → 201.36 | 2873.79 (identical) |
| jul | 297.23 | 297.33 | 241.73 → 241.86 | 1227.62 (identical) |
| jul_fwd | 308.66 | 308.72 | 250.24 → 250.31 | 1282.18 (identical) |
| forward | 217.35 | 217.23 | 196.93 → 196.80 | 980.73 (identical) |
| **overall** | **273.72** | **273.68** | | |

`prc.compare`: K=0 −0.04 [−0.14, +0.06], P(B better) 0.790. K=20 −0.05 [−0.17, +0.07]. K=200 −0.07 [−0.18, +0.05].
K=2000 −0.12 [−0.20, −0.03]. Month bootstrap −0.04 [−0.13, +0.07]. **VERDICT: REJECT**, "no difference on all rows".

- **Lane split.** Matched −0.05 [−0.18, +0.07]. Unmatched +0.00 (identical). All rows except LIRF −0.01 [−0.09, +0.08].
- **Per calendar month** (all rows / matched):
  - 2025-01: −0.09 [−0.27, +0.08] / −0.16 [−0.38, +0.09]
  - 2025-07: **+0.10** [−0.14, +0.32] / +0.12 [−0.18, +0.39]
  - 2025-11: −0.15 [−0.33, +0.03] / −0.17 [−0.36, +0.06]
  - 2025-12: −0.08 [−0.32, +0.13] / −0.09 [−0.31, +0.14]
- **Per airport** (all rows; ordered by share of squared disagreement):
  - LIRF −0.24 [−0.73, +0.25] (matched −0.37 [−1.07, +0.39]; 19.4 %)
  - LTFM −0.07 [−0.35, +0.23]
  - LFPG −0.00 [−0.25, +0.21]
  - EGLL −0.15 [−0.48, +0.20]
  - EHAM +0.04 [−0.22, +0.31]
  - EDDF +0.20 [−0.04, +0.46]
  - LEMD +0.04 [−0.19, +0.25]
  - EDDM +0.15 [−0.25, +0.54]
  - LEBL +0.08 [−0.17, +0.33]
  - LSZH **−0.38** [−0.71, −0.09]
- **Kill condition (a), LIRF lane-A RMSE:**
  - jan 282.93 → 282.33 (−0.60)
  - **jul 439.26 → 439.89 (+0.62)**
  - forward 299.30 → 298.08 (−1.22)
  - **jul_fwd +0.77** (LIRF all rows 735.22 → 735.98)
  - (a) **fails** twice, and "jul_fwd ≥ 0, close it" applies. (b) fails.

### 3.3 Reading

- The effect has the predicted sign and size, and it sits at LIRF (LIRF matched −0.80 and −0.37). It reverses in
  July on both configs. The note named July as the fold most likely to fail: LIRF's lowest-fill month, and C1's
  weak fold. The same pattern repeats here.
- The OSM model gains less than v16 (−0.04 vs −0.09). The note flagged the risk up front: the same size on the same
  instrument as C1, which did not transfer. At this size the ±0.5 s noise of adding any column to this model
  (`main_flseq.yaml` header) is the same order as the effect.
- Kill condition (c), the permuted twin run, was **not run**. Conditions (a) and (b) had already closed the item on
  both configs. The option exists (`fe_arr_fill_permute: true`) should anyone want the twin.

## 4. Hybrid

**Rule, written before any A1 number was seen** (scratch file dated 2026-09-30 04:34 UTC):

> The matched lane comes from `main_flseq_osm_nolirf_a1` if verdict(`main_flseq_osm_nolirf` →
> `main_flseq_osm_nolirf_a1`) is ADOPT, else from `main_flseq_osm_nolirf`. The unmatched lane comes from
> `main_flseq_a1` if verdict(`main_flseq` → `main_flseq_a1`) is ADOPT, else from `main_flseq`. Each lane is decided
> by the whole-run verdict of the run that supplies it. Model B does not get the A1 columns, so the unmatched lane
> is expected to be identical either way. The hybrid is then compared with the H2 OOF and reported whatever it reads.

Both verdicts are REJECT. The rule therefore selects `main_flseq_osm_nolirf` (matched) + `main_flseq` (unmatched),
which is **H2 itself**. The construction is the same as H2's (osm-on-flseq.md §8): lane = `mvt_minus_aobt3` non-null
in `data/work_flseq`. The rebuild script reproduces `outputs/hybrid_H2.parquet` exactly (`DataFrame.equals`) from the
two full-fit files.

- **hybrid_A1 vs H2 OOF:** identical on all 672,428 rows. The difference is +0.00 [+0.00, +0.00] at every K.
  **REJECT** (no difference).
- For the record only, and not selected by the rule: forcing A1 into both lanes (matched from
  `main_flseq_osm_nolirf_a1`, unmatched from `main_flseq_a1`, which is v16's model B) reads −0.04 [−0.14, +0.06]
  against H2. K=200 is −0.07 [−0.20, +0.04], and the month bootstrap is −0.04 [−0.13, +0.07]. That is **REJECT**.
  Only matched rows differ, so this is §3.2's matched-lane change. The K>0 figures differ slightly from §3.2,
  because H2's unmatched rows are v16's model B and so the drop set changes.

**File.** `outputs/hybrid_A1.parquet`: 344,841 rows (339,551 matched, 5,290 unmatched).

- Schema `{MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}`, equal to `data/raw/submitting.parquet`.
- Ids and order equal the template.
- 0 nulls and 0 negatives; min 0, max 113,641.
- `python -m prc.validate_submission` prints OK, with only the usual >10,800 s warning.
- Content equals `outputs/hybrid_H2.parquet` row for row. The file bytes differ only in writer metadata.
- No ADS-B override is applied. The v20 equivalent would be `scripts/adsb_override_candidate.py` on it, which gives
  v20 again.
- The A1 full-fit files `outputs/main_flseq_a1.parquet` and `outputs/main_flseq_osm_nolirf_a1.parquet` also pass
  the same checks (344,841 rows, template schema, ids and order, 0 nulls, 0 negatives).
- Nothing was submitted.

Logs (git-ignored):
- `logs/main_flseq_a1.log`
- `logs/main_flseq_osm_nolirf_a1.log` (the OOM attempt is `_oom.log`)
- `logs/compare_a1_flseq.txt` and `logs/compare_a1_flseq_detail.txt`
- `logs/compare_a1_osm_nolirf.txt` and `logs/compare_a1_osm_nolirf_detail.txt`
- `logs/compare_oof_hybrid_A1.txt` and `logs/compare_oof_hybrid_A1_forced.txt`
