# Item 20 — `v4-config-not-reproducible`

Investigator scope: code / config / git history only. No model was trained, no data frame was loaded.
Script: `docs/investigation_2026-09-16/scripts/v4-config-not-reproducible.py` (resolves the two YAMLs through
the two `TrainConfig` versions; the v4-era class is extracted from `git show 9f31cef:src/prc/train.py` with `ast`).

## 1. Hypothesis as stated (plan.md §1 item 20, source P 1)

`configs/submitted_v4.yaml` on today's code does not build v4: model B silently gets 255 leaves / min_data 20 /
3000 rounds (TrainConfig defaults), `hour_utc` is present, `force_row_wise` is on.

## 2. Procedure executed

```
git log --date=iso --format='%h %ad %s'                      # locate v4-era HEAD
git diff --stat 9f31cef d7073bf ; git diff --stat 9f31cef 71bf33b
cat configs/submitted_v4.yaml configs/main.yaml
git show a96e43a:configs/v4_linear_tree.yaml ; git show 9f31cef:configs/v4_linear_tree.yaml
git show 9f31cef:src/prc/train.py ; git show 9f31cef:src/prc/pipeline.py ; git show 9f31cef:src/prc/cv.py
git show 9f31cef:experiments/pipeline.jsonl ; git show 9f31cef:experiments/experiments.jsonl
git show d7073bf:experiments/pipeline.jsonl ; git show 4e807b5^:experiments/pipeline.jsonl
git log --all -S'v4_linear_tree-' -- experiments/ docs/      # any CV/pipeline record of a v4 run
git log -S'<ident>' -- src/prc/ configs/   for for_unmatched, unmatched_num_leaves, unmatched_min_data,
        unmatched_num_rounds, bagging_seed, force_row_wise, WX_KEY, hour_utc, linear_tree, scheme
git log -S'unmatched_num_leaves: int = 255' -- src/prc/train.py ; git log -S'unmatched_num_leaves: int = 31' ...
git show 0305847 d182fc1 70ef994 957c913 f5b950f a00b440 34808da ea04ec9 5b90919   (train.py / configs hunks)
git diff 9f31cef HEAD -- src/prc/ (stat) ; features.py / weather.py alias-name diffs ; train.py / predict.py code diffs
python3 experiments/pipeline.jsonl, experiments/submissions.jsonl  (read as JSON, printed)
.venv/bin/python docs/investigation_2026-09-16/scripts/v4-config-not-reproducible.py
```

## 3. Results

### 3.1 Which commit produced v4

| fact | value | source |
|---|---|---|
| v4 upload time | 2026-09-11T00:42:59Z, `resilient-hedgehog_v4.parquet`, LB 305.027 | `experiments/submissions.jsonl` version 4 (backfilled row) |
| last code commit before the upload | `9f31cef` 2026-09-10 10:16:39Z "feat: v2 on real data…" | `git log` |
| next commit | `d7073bf` 2026-09-11 01:55:54Z, docs only (`docs/03_prior_art.md`, 65 insertions) | `git diff --stat 9f31cef d7073bf` |
| commit after that | `71bf33b` 2026-09-13, docs + leaderboard JSON only | `git diff --stat 9f31cef 71bf33b` |
| config file used | `configs/v4_linear_tree.yaml` (renamed to `configs/submitted_v4.yaml` in `a00b440`, 2026-09-14, similarity 96 %, one line removed: `scheme: fast`) | `git show a00b440 --stat`, `git show 9f31cef:configs/v4_linear_tree.yaml` |
| run record of the v4 build | **none exists in any commit**: `experiments/pipeline.jsonl` at 9f31cef, d7073bf and 4e807b5^ holds v1, v2 (then v5/v6/v7); `experiments/experiments.jsonl` at 9f31cef holds v1, v2 only; `git log --all -S'v4_linear_tree-'` returns nothing. `outputs/` on disk has v1, v2, v5, v6, v7 predictions but no `v4_linear_tree.parquet`. `docs/07_experiments.md` @957c913 lists v3/v4 CV columns as "—". | git history, `ls outputs/` |

Consequence for (b): the v4-era *code path* and *config* are fully recoverable; the actual `rounds_matched` /
`rounds_unmatched` of the v4 build, its `n_features`, and the LightGBM/polars versions in use on 09-10/11 are not
recorded anywhere in the repository (the only pinning is `lightgbm>=4.5`, `polars>=1.30`; installed today: lightgbm
4.7.0, polars 1.44.2; no lock file).

### 3.2 Resolved configuration table

Loader then (`pipeline.py:123` @9f31cef): `TrainConfig(**{k: v for k in spec if k in __dataclass_fields__})`.
Loader now (`pipeline.py:128-137`): same filter, plus `params` merge and list→tuple casts. Both silently drop
`note` and `use_weather` (the latter is read separately from `spec`). Today's loader also silently drops `scheme`
if present (field removed in `a00b440`); `submitted_v4.yaml` no longer carries it.

Model B at 9f31cef: `predict_fold` called `_fit(u_tr, …, cfg, cols_b, …)` with the **shared** cfg (train.py
@9f31cef, unmatched branch). Model B today: `_fit(u_tr, …, cfg.for_unmatched(), …)` (train.py:358-360).

| field | v4-era (`v4_linear_tree.yaml` @9f31cef, code 9f31cef) | `submitted_v4.yaml` on HEAD (f250a9f) | differs |
|---|---|---|---|
| A.num_leaves / min_data_in_leaf | 127 / 50 | 127 / 50 | |
| A.num_rounds cap / early_stopping | 1500 / 100 | 1500 / 100 | |
| A.linear_tree / linear_lambda | True / 1.0 | True / 1.0 | |
| A.feature_fraction / bagging_fraction / bagging_freq / lambda_l2 / lr | 0.8 / 0.8 / 1 / 1.0 / 0.05 | same | |
| A.deterministic | True | True | |
| A.force_row_wise | **absent** | **True** | yes |
| A.seed (early-stop fit) | 42 | 42 | |
| A refit params | `params` + `seed=42` (one fit) | `params` + `seed=42, bagging_seed=42, feature_fraction_seed=42` (one fit, n_seeds=1) | yes |
| **B.num_leaves** | **127** | **255** | yes |
| **B.min_data_in_leaf** | **50** | **20** | yes |
| **B.num_rounds cap** | **1500** | **3000** | yes |
| B.linear_tree / linear_lambda | True / 1.0 | True / 1.0 | |
| B.force_row_wise | absent | True | yes |
| B refit params | `params` + `seed=42` | + `bagging_seed=42, feature_fraction_seed=42` | yes |
| n_seeds | (no field; one fit) | 1 | equivalent count |
| use_weather (spec) | True | True | |
| weather join key | `hour_utc` → **overwrites and drops the `hour_utc` feature** | `_wx_hour` (`WX_KEY`) → `hour_utc` feature **kept** | yes |
| use_encodings | (no field; always on) | True | equivalent |
| `op_offset_*` encoding | did not exist | fitted but hidden (`use_offset_encoding=False`) | equivalent for the matrix |
| CANDIDATE_COLS (9) | did not exist | computed, hidden (`include_candidates=()`) | equivalent for the matrix |
| CATEGORICALS | 10 (incl. ADES_mvt, ADES_REGION) | 10, same list | |
| inner_holdout_months / reference_min_count | 1 / 30 | 1 / 30 | |
| CV folds | jan, jul, forward (`scheme: fast`) | jan, jul, jul_fwd, forward | CV only; not the ranking model |

`configs/main.yaml` for comparison (today): A 127/50, rounds 2000, no linear_tree, 3 seeds, force_row_wise;
B 127/50/2000 (explicit `unmatched_*` keys), weather on.

### 3.3 Exact difference list — "v4 as scored" vs "submitted_v4.yaml run today", with commits

1. **Model B num_leaves 127 → 255, min_data_in_leaf 50 → 20, rounds cap 1500 → 3000.**
   `for_unmatched()` introduced in `0305847` (2026-09-14 01:43) with defaults 31/40/600; defaults changed to
   255/20/3000 in `957c913` (2026-09-14 05:20). `submitted_v4.yaml` has no `unmatched_*` keys, so the defaults
   apply. (`f5b950f` 2026-09-14 09:57 reverted **main.yaml** to 127/50/2000 but left the dataclass defaults at
   255/20/3000; `submitted_v4.yaml` was created in `a00b440` between those two commits and was not touched.)
   Note: the v4-era model B cap was 1500 (the shared `num_rounds`), not 2000 as in `repro_v2.yaml`.
2. **`force_row_wise: True` added to the shared params** for both models — `70ef994` (2026-09-15 06:56).
3. **Refit seeds**: the final refit now passes `bagging_seed` and `feature_fraction_seed` explicitly (equal to
   `seed`), for both models — `d182fc1` (2026-09-14 04:28), `_fit` loop (train.py:301-302). At 9f31cef the refit
   passed only `seed`. The early-stopping fit is unchanged in both versions (`params` only).
4. **`hour_utc` present in the design matrix**: the weather join key was renamed `hour_utc` → `_wx_hour`
   (`WX_KEY`) in `34808da` (2026-09-15 07:49). At 9f31cef `join_weather` overwrote the integer `hour_utc`
   feature with a truncated timestamp and then dropped it, so v3/v4 trained without `hour_utc` (both models;
   `hour_local` is derived earlier and survives). Same commit also changed the time-zone handling of the join
   (the METAR side is now converted to the departures' tz); at 9f31cef the join ran on whatever dtypes the
   cached METAR table had at the time — the commit message says the join "could not run at all" on the current
   data, so the v4-era join behaviour on today's data cannot be re-executed as-is.
5. **Encodings**: a fourth table `op` (`operator_offset_table`) is fitted and joined (`e5af4b5` 2026-09-15
   10:25) but its two columns are hidden unless `use_offset_encoding: true`; not in the matrix under
   `submitted_v4.yaml`. Nine candidate columns are computed (`66c088b`, `30e21e2`, `e5af4b5`, `d75c9f9`,
   `5794f24`) and hidden by `f9fff60` / `97e67c6` (opt-in). Matrix-neutral; feature-cache and wall-clock only.
6. **Weather column set**: unchanged (alias-name diff of `weather.py` 9f31cef→HEAD is empty; 18 columns).
7. **Predict path**: `predict_ranking` (predict.py) is textually identical apart from docstrings; `write_submission`
   gained a `reference_median` gate (`d5c4535`). `_assemble` is a factored-out copy of the old tail of `predict_fold`.
8. **CV folds**: `scheme` field removed (`a00b440`), `jul_fwd` fold added (`16c910e`), `folds:` filter added
   (`9ab8758`). Affects the CV record only, not the ranking model.
9. **Not derivable from the repository**: LightGBM / polars / numpy versions on 2026-09-10/11 (unpinned, no lock),
   the METAR cache contents at that time (`data/ref/metar_hourly.parquet` not versioned), and the v4 build's
   actual round counts.

Commit index for the items the task asked to record:

| what | commit(s) |
|---|---|
| `for_unmatched()` introduced | `0305847` (2026-09-14 01:43) |
| `for_unmatched()` call site changed | `5b90919` (b_cfg for split_mode), `ea04ec9` (reverted to plain `cfg.for_unmatched()`), `70ef994` (comment only) |
| `unmatched_*` defaults | `0305847` → 31/40/600; `957c913` → 255/20/3000 (still current) |
| `unmatched_*` in main.yaml | `a00b440` 255/20/3000 → `f5b950f` 127/50/2000 |
| `bagging_seed` / `feature_fraction_seed` | `d182fc1` (only commit that ever touched them) |
| `force_row_wise` | `70ef994` (only commit) |
| weather join key `hour_utc` → `_wx_hour` | `34808da` |
| `scheme` field removed | `a00b440` |
| `v4_linear_tree.yaml` → `submitted_v4.yaml` | `a00b440` |

## 4. Verdict per sub-claim

| sub-claim | verdict | basis |
|---|---|---|
| `submitted_v4.yaml` on today's code gives model B 255 leaves / min_data 20 / 3000 rounds | **CONFIRMED** | script output; `for_unmatched()` + dataclass defaults (train.py:93-95); YAML has no `unmatched_*` keys |
| v4 as scored used 127 / 50 for model B | **CONFIRMED** | `_fit(u_tr, …, cfg, …)` at 9f31cef uses the shared params |
| v4 as scored used 1500 rounds cap for model B | **CONFIRMED** (P 1 wrote "127/50/1500"; correct) | shared `num_rounds: 1500` |
| `hour_utc` present today, absent in v4 | **CONFIRMED** (code reading) | `join_weather` @9f31cef overwrote/dropped `hour_utc`; `WX_KEY` since `34808da` |
| `force_row_wise` on today, absent in v4 | **CONFIRMED** | `70ef994`; script side-by-side |
| refit seed semantics differ (explicit `bagging_seed`/`feature_fraction_seed`) | **CONFIRMED** as a code difference | `d182fc1`; whether it changes the booster is item 21, not re-tested here |
| model A hyper-parameters resolve identically | **CONFIRMED** except `force_row_wise` and the refit seed keys | script |
| "v4 is not reproducible from `submitted_v4.yaml` today" | **CONFIRMED** at the config level (five resolved differences: B leaves, B min_data, B rounds cap, force_row_wise, hour_utc; plus refit seed keys) | §3.3 |
| "v4 is reproducible at all" (with any config on today's code) | **UNDETERMINED** | no config key restores the `hour_utc`-dropping join or the absence of `force_row_wise`; `exclude_features: [hour_utc]` + `unmatched_*: 127/50/1500` covers items 1 and 4 but not 2, 3, library versions or METAR cache state |
| the v4 build's own rounds / n_features | **UNDETERMINED** | never logged (§3.1) |

## 5. What cannot be determined without a run

- Whether the five resolved differences move the ranking predictions, and by how much; in particular whether
  `submitted_v4.yaml` today reproduces v4's 305.03 or lands near v5's 325.22 (same B settings 255/20/3000, but v5
  had no linear_tree and no weather).
- Whether `force_row_wise` alone changes model A's or B's rounds (item 21/28 territory).
- Whether the absence of `hour_utc` is what drove v3/v4's model B behaviour (item 22).
- Whether today's LightGBM 4.7.0 reproduces the 09-10 booster bit-for-bit (version at the time unknown).

## 6. Facts vs inference

**Facts** (all from git objects, files on disk, or the script's output): everything in §3.1–§3.3 and the
commit index; the verdict table's "basis" column.

**Inference** (mine, not measured): (i) that 9f31cef is the exact tree the v4 run executed — supported by the
timestamps (last code commit 14 h before the upload, next commit docs-only 1 h after) but the working tree at
run time could have carried uncommitted edits; nothing in the repo can exclude that. (ii) That items 5–8 in
§3.3 are matrix-neutral — follows from `visible_numeric_columns()` hiding CANDIDATE_COLS/OFFSET_COLS, and from
`predict_ranking` being unchanged, but was not verified by building a matrix. (iii) That the weather column set
was identical — inferred from an empty alias-name diff of `weather.py`; the METAR cache content is not versioned.
