# Hand-off: the R6 base, for the v36–v38 chain on `claude/laughing-cray-vvwh0k` (2026-10-05)

From `claude/model-b-and-arrival-features`. These are facts only, with paths and commits. Nothing here was trained or submitted.

## 1. What R6 is

- **Definition:** `docs/competitor_research_2026-09-21/results/rejected-rereview.md`.
  - §3 was pre-registered in **23a30d9**. The result and the choice are in §4, commit **1d6a33c**.
  - Matched lane: sum-to-one least-squares weights over five members, fitted leave-one-fold-out (LOFO) with `catboost_blend.sum_to_one_weights`.
  - Unmatched lane: `u` = `main_flseq`.
- **Members** (exp ids are from `scripts/blend_mlp.py`):

| key | exp id | full-fit file (`blend_mlp.FULL_FILES`) |
|---|---|---|
| a2o | `main_flseq_osm_nolirf_l2_opt-7244b3004f` | `outputs/main_flseq_osm_nolirf_l2_opt.parquet` |
| pao | `main_flseq_osm_nolirf_l2_pa_opt-8d54effc06` | `outputs/main_flseq_osm_nolirf_l2_pa_opt.parquet` |
| p2o | `main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c` | `outputs/main_flseq_osm_nolirf_plainy_l2_opt.parquet` |
| c2 | `main_flseq_osm_nolirf_cb_l2-4aa341b206` | `outputs/main_flseq_osm_nolirf_cb_l2.parquet` |
| m4 | `main_flseq_osm_nolirf_l2_mlp-2b9a3b9348` | `outputs/main_flseq_osm_nolirf_l2_mlp.parquet` |
| u (unmatched) | `main_flseq-8615e6a194` | `outputs/main_flseq.parquet` |

- **Verdict against `hybrid_blend_l2_pa_opt_mlp`:** `prc.compare` on 672,428 rows.
  - Full: **−0.32 s [−0.44, −0.24]**.
  - Drop-200: **−0.36 s [−0.48, −0.26]**.
  - Month bootstrap: −0.32 [−0.38, −0.23]. Verdict ADOPT.
- **Weights fitted on all OOF folds** (these are the ones used in the full-fit file):
  - a2o 0.069428
  - pao 0.374785
  - p2o 0.176099
  - c2 0.147002
  - m4 0.232686
- **Per-fold weights** (held-out fold: a2o / pao / p2o / c2 / m4):
  - forward: 0.133 / 0.345 / 0.143 / 0.134 / 0.246
  - jan: 0.069 / 0.360 / 0.181 / 0.149 / 0.240
  - jul: 0.015 / 0.422 / 0.196 / 0.156 / 0.211

## 2. The full-fit file: `outputs/hybrid_blend_reweight.parquet`

- **Status:** it now exists on this container's disk. It was built at 11:49 UTC, after 1d6a33c (log: `logs/rereview/full.log`).
  - md5 `c4cb123d88ad427921bc506fc9c7e3b5`.
  - 344,841 rows, of which 339,551 are matched and 5,290 unmatched.
  - The schema, ids and order equal the template. 0 nulls and 0 negatives. Max 113,641.
  - Against `hybrid_blend_l2_pa_opt_mlp` it differs on 322,153 rows (RMS 15.64 s, max |d| 862). The unmatched rows are untouched.
  - The status line at the top of `rejected-rereview.md` and its §4 still say "full-fit file not built". They were written before the build and are stale.
- **It is not in git.** `outputs/` is git-ignored (`.gitignore:31`).
- **To rebuild it** (no training; it reads only the six full-fit member files above), run this from the repo root:

```
POLARS_MAX_THREADS=1 PYTHONPATH=src:docs/competitor_research_2026-09-21/scripts:scripts \
  .venv/bin/python docs/competitor_research_2026-09-21/scripts/rejected_rereview.py full \
  --weights '{"a2o": 0.069428, "pao": 0.374785, "p2o": 0.176099, "c2": 0.147002, "m4": 0.232686}'
```

  - The default `--out` is `outputs/hybrid_blend_reweight.parquet`.
  - Matched rows are `rint(Σ w·member)`. Unmatched rows are `outputs/main_flseq.parquet`. Lane = `mvt_minus_aobt3` non-null in `data/work_flseq/feat_rank.parquet`.
  - It also needs `data/raw/submitting.parquet`.
  - The script inserts `scripts/` into `sys.path` itself. `blend_mlp`, `catboost_blend` and `plainy_blend` are all in `scripts/`. The PYTHONPATH above covers both script directories and `src`.
  - The six member files are git-ignored. The other session can rebuild only if its container already has the same six `outputs/` files.

## 3. The R6 OOF: `logs/rereview/oof_rr_R6_pa5_lofo.parquet`

- **Status:** on this container's disk, md5 `80517938aec67b1fcd4ac1a0e71b36e7`. It has columns `MVT_ID_mvt` and `oof` and 2,085,047 rows. Log: `logs/rereview/r6.log` (R6 RMSE 267.2520).
- **Not in git:** `logs/` is git-ignored (`.gitignore:35`).
- **Coverage by month:**
  - Non-null on every row of 2025-01, 2025-07, 2025-11 and 2025-12.
  - Null for Feb–Jun and Aug–Oct 2025.
  - The Nov+Dec values are the forward-fold OOFs of all five members, combined with the held-out-forward LOFO weights.
- **To rebuild it** (deterministic: lstsq on fixed OOFs):

```
POLARS_MAX_THREADS=1 PYTHONPATH=src:docs/competitor_research_2026-09-21/scripts:scripts \
  .venv/bin/python docs/competitor_research_2026-09-21/scripts/rejected_rereview.py r6 --scratch DIR
```

  - It reads `data/work_flseq/feat_train.parquet`, plus `experiments/oof_<id>.parquet` for every entry of `ARMS` in the script.
    - That is the six ids above, plus `main_flseq_l2b-e40f03b86d`, `main_flseq_osm-bb204cd922` and `main_flseq_osm_nolirf_l2_lt-a9fb149ab5`.
    - `frame()` joins all of them, even though R6 uses only the five members and `u`.
  - It also reads a2o's `_folds` sidecar.
  - All of these are git-ignored (`*.parquet`).

## 4. The learned ADS-B residual, refit on R6

- **Where:** `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md` §13, commit **3f76f8e**.
- **What changed:** the §12 design is kept and only the base is swapped. The commands are `scripts/adsb_learn_cv.py --base r6` and `scripts/adsb_learn_apply.py --base r6`. `b` is the R6 OOF above (`adsb_learn_cv.R6_OOF`).
- **CV gain on 344,419 Jan+Jul rows:** 7,928.5 s² [6,927.8, 8,980.8]. RMSE goes 311.762 → 298.776. The final fit uses 325 rounds.
- **(R6 + learned) − (mlp + learned): −0.158 s [−0.343, +0.017].** The CI includes 0.
- **Files.** Both are on local disk only and git-ignored, and neither is submitted.
  - `outputs/hybrid_blend_reweight_adsb_learned.parquet`: R6 + learned, with c = 0 at LTFM.
  - `outputs/v36_candidate_reweight_adsb_posjoin.parquet`: the above + the unchanged 430-row positional overlay (`analysis/local_posjoin_overlay.py` at bdf0b91).
  - The residual model is trained on Jan+Jul 2025 only.

## 5. Finding from this session's reproduction attempt: the v37/v38 swap cannot be applied as-is to an R6 file

- **How the swap works.** `analysis/local_adsb_apply_months.py` (v36/C2 → v37) and `analysis/local_adsb_apply_swap.py` (v37 → v38 and C4), on `claude/laughing-cray-vvwh0k`:
  - They recover `b` from the input file by the fixed point `b = file − c_OLD(b)`.
  - They write `file + c_NEW(b) − c_OLD(b)` on ground-fix rows.
  - So they assume that file = b + c_OLD, where c_OLD is a v35-style residual: one fitted on residuals against their base OOF.
  - Their training residuals are `y − b`:
    - Jan+Jul rows use the v23-equivalent cb-hybrid OOF.
    - Nov+Dec rows use `main_flseq_osm_nolirf_fw` (`docs/months_2026-10-05/README.md`).
- **Consequence for R6.** An R6-based file is R6 + c_R6-learned (+ overlay). Rebasing on R6 needs the swap's c_OLD and c_NEW recomputed with R6 as `b`: both in the training residuals and in the recovery of `b`. The v37/v38 deltas cannot be applied as they are.
- **The posjoin overlay depends on the base too.** The v36 posjoin learned-residual delta depends on the base value, so it also has to be recomputed on R6.
- **What lives only in their container:**
  - the Nov+Dec 2025 ADS-B features (`feat_2025novdec.parquet`)
  - the Aug–Oct features (`feat_2025aso.parquet`)
  - the forward-fold and `aso`-fold OOFs (`main_flseq_osm_nolirf_fw`, `main_flseq_fw`, `main_flseq_osm_nolirf_aso`, `main_flseq_aso`)
  - the candidate files `c1_…`–`c4_…`
  - None of these is on this container or in git.
- **Usable from this side.** The R6 OOF of §3 has five-member forward-fold values for Nov+Dec 2025. It could serve as `b` for their 4-month residual. It has no Aug–Oct rows, so the 7-month variant would still need their `aso` OOF.

## 6. Committed versus local only

| item | where |
|---|---|
| R6 definition, verdict, weights | git: `rejected-rereview.md` (23a30d9, 1d6a33c) |
| Build code (`r6`, `full`) | git: `docs/competitor_research_2026-09-21/scripts/rejected_rereview.py`; `scripts/blend_mlp.py` (`FULL_FILES`) |
| ADS-B refit on R6, results | git: `adsb-learned-residual.md` §13; `scripts/adsb_learn_{cv,apply,features}.py` (3f76f8e) |
| `outputs/hybrid_blend_reweight.parquet` | local disk only (git-ignored) |
| `logs/rereview/oof_rr_R6_pa5_lofo.parquet`, `full.log`, `r6.log` | local disk only (git-ignored) |
| `outputs/hybrid_blend_reweight_adsb_learned.parquet`, `outputs/v36_candidate_reweight_adsb_posjoin.parquet` | local disk only (git-ignored) |
| The six member full-fit files and the `experiments/oof_*.parquet` inputs | local disk only (git-ignored) |
