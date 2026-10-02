# LightGBM `linear_tree` (`linear_lambda` 15) in the LGB members of hybrid_blend_l2 (2026-10-02)

**Verdict:** pending (§0 is the pre-registration; results go in §1 onward, §0 is not edited after the first run).

## 0. Pre-registration (written before any run)

### 0.1 Why, and what the public evidence is

`public-code-deep-dive-2026-09-30.md` §2.2 and §4 row 4 [S]: kind-mango's base members are LightGBM with
`linear_tree`; their "MB5 clock-era" Optuna study (v76) set `linear_lambda` 15.09 together with lr 0.01514,
613 leaves, `min_data_in_leaf` 515, `feature_fraction` 0.837, bagging 0.909 / 5, `lambda_l1` 0.002,
`lambda_l2` 0.027, `min_gain_to_split` 0.348, 3,377 rounds. Board: v76 265.27, −0.88 s (−465 MSE); hold-out −965
MSE for the member, −240 in the blend. That −0.88 s is the **retune** (linear_tree was already on since their v21,
−15.6 s bundled, raw y). joyous-rainbow reports `linear_tree` alone as −31.4 s on raw y. Neither number isolates
`linear_lambda` 15 on an anchored target. No competitor code was opened for this note.

This note tests only the `linear_tree` + `linear_lambda` 15 part. **The Optuna retune is out of scope; nothing is
tuned.**

### 0.2 What we measured before (quotes)

- `docs/investigation_2026-09-16/results/linear-tree-model-b.md` §4.1 (item 11; `configs/main.yaml` era, forward
  fold, `linear_lambda` 1.0, linear_tree in **both** models): `rounds_matched` 939 → 231, "RMSE matched
  (n = 325,004) 201.6633 → 202.5132 (+0.85)", unmatched 2,109.32 → 1,181.29. `findings.md`: full −53.58
  [−96.86, −17.04]; K=200 +1.22 [+0.75, +1.62] → "REJECT — tail-only".
- `model-a-ceiling.md` §5.3: at 435,263 rows, 31 / 127 / 511 leaves give matched-lane RMSE 213.229 / 213.125 /
  213.376 — "A 16× range of model capacity moves the answer by 0.25 s, while a 2× range of training rows moves it
  by 5.8 s."
- `plans/P7-model-capacity.md` §1.2: "The 2000 cap does not bind on model A. Early stopping selects ~940–1,260
  rounds at lr 0.05 / 127 leaves." P7 itself was never run (no P7 record in `experiments/experiments.jsonl`;
  `EXECUTION-ORDER.md` item 9: "8 h of compute … if it competes with P4 for compute, P7 loses").
- `src/prc/train.py` (model B comment): "31 leaves is worse than 127, 255 is better than 127, 511 adds nothing."

### 0.3 Compatibility with our training code (checked before writing this)

- **Params pass-through.** `use_linear_tree` and `linear_lambda` are existing `TrainConfig` fields;
  `lgb_params()` adds `linear_tree: True` and `linear_lambda` when the flag is on. `_fit` passes those params to
  every `lgb.Dataset` at construction (`params=params`) and to `lgb.train`, which `linear_tree` requires (the raw
  feature values must be kept at Dataset construction). A recipe's `params:` block is merged over the defaults
  (`train_config_from_spec`), so nothing else moves.
- **Model B.** `use_linear_tree` is one field and `for_unmatched()` copied it, so model B would get linear trees
  too — the confound of item 11 (`linear-tree-model-b.md` §2). New flag **`linear_tree_matched_only`** (default
  off, out of the hash at off): on, `for_unmatched()` turns `use_linear_tree` off, so model B is fitted with
  exactly the base's parameters and its predictions should be byte-identical. Only model A's matched lane enters
  the blend, so this isolates the thing being tested. Refused without `use_linear_tree`.
- **Categoricals.** LightGBM 4.7.0, probed on a toy set: a feature passed in `categorical_feature` is used for
  splits but **never enters a leaf's linear model** (`leaf_features` never contains it). Our ten categoricals are
  integer codes passed that way, so they stay split-only.
- **NaNs.** Probed the same way: a leaf's linear model uses the numeric features split on along its branch; a row
  with NaN in any of them gets the leaf's constant (`leaf_value`), not the linear output. LightGBM also drops such
  rows from the leaf's least-squares fit. Our matrix keeps nulls as NaN (float32, never 0-filled), which is what
  LightGBM requires. So linear leaves are well defined but act only on rows complete in that leaf's features.
  No ±inf in any of the 105 float columns of `data/work_flseq/feat_train.parquet`.
- **Scaling.** Features are not standardised (seconds up to 86,400 next to counts and km), so `linear_lambda` 15
  shrinks coefficients unevenly across features. Recorded, not changed: the arm copies the public setting.
- **First tree** is a constant (boost-from-average); linear leaves start from tree 2.
- **Memory.** `linear_tree` keeps raw feature values inside each LightGBM Dataset; expect a higher peak than the
  anchored+L2 run's 11,231 MB (`feature-levers-L1-L4.md` §6).

### 0.4 Arms and recipes

| arm | recipe | exp id | differs from its base by |
|---|---|---|---|
| base | `configs/main_flseq_osm_nolirf_l2.yaml` | `main_flseq_osm_nolirf_l2-ef204b44ff` (exists) | — |
| **LT** | `configs/main_flseq_osm_nolirf_l2_lt.yaml` | `main_flseq_osm_nolirf_l2_lt-a9fb149ab5` | `use_linear_tree: true`, `linear_lambda: 15.0`, `linear_tree_matched_only: true` |
| plain-y LT (only if LT is ADOPTED) | `configs/main_flseq_osm_nolirf_plainy_l2_lt.yaml` | `main_flseq_osm_nolirf_plainy_l2_lt-af11759d5d` | the same three keys over `main_flseq_osm_nolirf_plainy_l2` (`c74a27239b`) |

Everything else is unchanged: lr 0.05, 127 leaves, `min_data_in_leaf` 50, `feature_fraction` 0.8, bagging 0.8 / 1,
`lambda_l2` 1, 2,000-round cap, early stopping 100 on the one-month inner holdout, 3 seeds, the four folds
(jan, jul, jul_fwd, forward; pooled jan / jul / forward = 672,428 rows). `tests/test_linear_tree.py` checks the
three-key diff, that model B's resolved parameters equal the base's, and that every existing recipe keeps its id
and its resolved LightGBM parameters for both models (byte-identical, computed at 23018be).

### 0.5 Decision rule

`prc.compare.verdict`: the full CI (K=0) and the drop-200 CI must both exclude zero on the same side; ADOPT needs
the improving side.

**Step 1 — member.** `python -m prc.compare main_flseq_osm_nolirf_l2-ef204b44ff main_flseq_osm_nolirf_l2_lt-a9fb149ab5
--work-dir data/work_flseq`. Reported: K = 0 / 20 / 200 / 2000 curve, month bootstrap, and
(`scripts/blend_l2.py detail --scratch experiments`, read-only) the lane split (matched / unmatched / all except
LIRF), per calendar month and per airport. Expected and checked: unmatched lane +0.00 (byte-identical model B).
- If LT is **not** ADOPTED: stop. No plain-y run, no blend, no file.
- If LT is ADOPTED: run plain-y LT (CV + full fit) and the anchored LT full fit, then step 2.

**Step 2 — blend.** Candidate `hybrid_blend_l2_lt`: matched lane = equal-weight mean of anchored LT, plain-y LT and
the unchanged CatBoost+L2 (`main_flseq_osm_nolirf_cb_l2-4aa341b206`); unmatched lane = `main_flseq-8615e6a194`'s
model B (unchanged). Base = `hybrid_blend_l2_P` (the OOF of `outputs/hybrid_blend_l2.parquet`; its rebuild must
reproduce `blend-l2.md`'s RMSE 269.98). Weights fixed at 1/3. OOF hybrids are written by `scripts/blend_lt.py oof`
to a scratch directory and compared with `prc.compare --exp-dir <scratch>`, with the same detail tables.
Information only, not the decision: the plain-y LT member vs plain-y+L2 (matched lane swapped, same unmatched lane).
- If ADOPTED: build `outputs/hybrid_blend_l2_lt.parquet` (`scripts/blend_lt.py full`): matched rows
  `rint((a + p + c) / 3)` from the three Int32 full-fit files, unmatched rows from `outputs/main_flseq.parquet`;
  validate rows (344,841), schema, ids and order against `data/raw/submitting.parquet`, no nulls, no negatives.
- If not ADOPTED: keep `hybrid_blend_l2`; no file.

Nothing is uploaded or submitted, and no ADS-B residual is applied.

### 0.6 Runs and resources

`.venv/bin/python -m prc.pipeline --config configs/<recipe>.yaml --work-dir data/work_flseq`, one at a time,
nothing else training; logs to `logs/linear_tree/` (git-ignored), exit codes checked. Step 1's run is CV only
(`--skip-predict`); its full fit (`--skip-cv`) is run only if LT is ADOPTED. Peak cgroup `total_rss` is sampled
every 10 s from `memory.stat` of the cgroup `/proc/self/cgroup` names and of its parent (limit 13,681 MB). **If
`total_rss` exceeds 13,000 MB the run is killed and reported, not retried.**
