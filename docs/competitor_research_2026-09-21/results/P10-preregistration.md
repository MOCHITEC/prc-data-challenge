# P10 pre-registration — CatBoost vs LightGBM (written 2026-09-28, before any competition-data fit)

Written after Stage 0 (plumbing + synthetic tests) and **before** the first fit on competition data —
slice or full. Its SHA-256 and write time are recorded in `results/p10-runs/prereg.sha256`. This file is not
edited after that; anything learned later goes in `P10-catboost-measured.md` and says so.

## What is run

All arms are `configs/main_v15.yaml` (the v15 build recipe, `data/work` frame, 143 columns) with the same
three changes for both families: `n_seeds: 1`, `folds: [forward, jul_fwd]`, `es_holdout_encodings: false`.
`lgb_deterministic` at its default True. CatBoost at `CB_DEFAULTS`: depth 6, SymmetricTree, `learning_rate`
0.1, `max_ctr_complexity` 1 set explicitly, `thread_count` 4, `has_time` False (library default), same
`num_rounds` 2000 / `early_stopping` 100 / inner-holdout protocol / round-scaling formula / seed as LightGBM.

| arm | config | cell |
|---|---|---|
| L+E | `p10_lgb_e` | LightGBM, encodings on |
| L−E | `p10_lgb_ne` | LightGBM, encodings off |
| L−E′ (control) | `p10_lgb_ne_s43` | L−E at seed 43 — the refit redraw of one family on the same information |
| C−E | `p10_cb_ne` | CatBoost, encodings off |
| C+E | `p10_cb_e` | CatBoost, encodings on |
| C−E ctr4 | `p10_cb_ne_ctr4` | C−E at `max_ctr_complexity` 4, forward only |

Order: Stage 1 slice (forward fold, 20 % hash-subsample of training rows, full validation month, C−E and L−E,
plus a 5 % slice for memory scaling and a ctr4 slice for its wall clock) → the −E pair and the control on both
folds → the +E pair on both folds → C−E ctr4 on forward if the budget allows. One fitting process at a time.

**Learning-rate rule, fixed now.** CatBoost's own auto rule gives 0.073–0.076 for 2,000 iterations at
1.43–1.74 M rows (measured, synthetic, no eval set), i.e. it plans to use all 2,000. 0.1 is chosen so that early
stopping can fire under the shared cap. If a full-data CatBoost fit's early-stopping best count reaches the cap
(≥ 1,990 on model A), that arm is flagged **capped** and is re-run once at `learning_rate` 0.2 before it is
read. The trigger is the training-side round count, never a validation RMSE. No other CatBoost setting is tuned.

## How it is read

* The rule is `prc.compare.verdict`: full-data CI **and** K=200 CI both excluding zero on the same side.
  Read **pooled over [forward, jul_fwd]** and **per fold**, B = CatBoost, A = LightGBM, in each cell.
* **Lane A (matched) and lane B (unmatched) are reported separately.** Lane B is not used to decide anything:
  it is ~6,900 rows whose squared error is carried by a handful of rows.
* **The control prices the refit lottery.** L−E → L−E′ is the same model family at a different seed. A family
  difference is only read as one if its lane-A point estimate is larger in magnitude than **3×** the control's
  lane-A draw on the same fold, on both folds.
* **Shrinkage.** A family effect is a *result* only if it has the same sign on both folds and the smaller fold's
  point estimate is at least **one third** of the larger's. Otherwise it is a one-fold lead and is called that.
* Board pricing uses dBoard/dOOF = 0.747 (n = 1) on the pooled K=0 difference, and nothing else.

## What I expect (point guesses, so they can be wrong)

* **E1 (the P2-invariant cell).** C−E vs L−E pooled: **within ±5 s**, sign not called. P(CatBoost ADOPT) ≈ 0.35.
  Reason: the categorical block alone was worth 7.4 s to an older LightGBM (P10 §3.3), so ordered TS vs
  LightGBM's categorical splits cannot be worth much more than that without also winning on tree shape.
* **E2.** C+E vs L+E pooled: **−2 s to +6 s**, i.e. CatBoost no better with our encodings; the Greedy-TS columns
  are the prediction-shift CatBoost's ordered TS is designed to avoid, and CatBoost gets them anyway.
* **E3 (interaction).** The encodings are worth less to CatBoost than to LightGBM: (C+E − C−E) closer to zero
  than (L+E − L−E).
* **E4.** ctr4 vs ctr1 on forward: **|Δ| < 1 s**, at ≥ 2.5× the wall clock.
* **E5.** CatBoost peak memory on a full fold is higher than LightGBM's by 1–3 GB; wall clock 2–4× per fold.

## What stops me

* **M1 memory.** If the slice scaling predicts a full-fold CatBoost peak above 12,500 MB of the 13,681 MB limit,
  the configuration is reduced **identically for both families** (and the LightGBM arms re-run on it), or the
  arm is not run. Never a smaller CatBoost against a full-size LightGBM.
* **M2 slice.** CatBoost more than 30 % worse than LightGBM on the 20 % slice → a bug hunt, not a comparison.
* **S1.** C−E REJECT as "worse on all rows" pooled → no ctr4 arm; the −E contrast is the answer.
* **S2.** CatBoost ahead only in the −E cell (C−E beats L−E but C+E does not beat L+E) → the finding is about
  the encodings (P2), not about the family.
* **Determinism.** Two runs of the same CatBoost config on the slice must agree bit for bit, or nothing that
  follows is read.

## The best possible outcome of this session, stated now

**NEEDS-MORE, not SHIP.** Even a clean pooled ADOPT on two folds at one seed is exactly the evidence that
failed to transfer twice today (v15's dead-column removal, C1). SHIP would need the production fold set
(jan, jul, forward) at n_seeds 3 to agree and one board probe. **DO-NOT-SHIP** if CatBoost loses in the +E cell
(the one production uses), if the effect fails the shrinkage or control tests, or if S1/S2 fire.
