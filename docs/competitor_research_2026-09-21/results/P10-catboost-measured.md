# P10 — CatBoost against LightGBM: **DO-NOT-SHIP as run; ON HOLD, not closed**

Written 2026-09-28 from the arms recorded in `experiments/experiments.jsonl`. The executing agent was stopped
by a container restart after five of its arms had finished and before it wrote this note; the verdict below
applies the rules in `P10-preregistration.md` (SHA-256 in `p10-runs/prereg.sha256`, written before any
competition-data fit) to those recorded numbers, and nothing else. One arm, `p10_cb_ne_s43_jf` (CatBoost's own
seed control on `jul_fwd`), was still running when this was written and is appended in §4 when it lands.

All arms: `configs/main_v15.yaml` with `n_seeds: 1`, `folds: [forward, jul_fwd]`, `es_holdout_encodings:
false`, identical for both families. CatBoost at depth 6, SymmetricTree, `learning_rate` 0.1,
`max_ctr_complexity` 1, same round cap, early stopping, inner holdout and seed as LightGBM. Untuned by design.

## 1. The numbers (lane A = matched, the lane the pre-registration decides on)

| arm | `jul_fwd` full | `jul_fwd` lane A | `forward` full | `forward` lane A | lane B fwd | rounds fwd |
|---|---:|---:|---:|---:|---:|---:|
| L−E `p10_lgb_ne` | 316.3631 | 254.072 | 222.4007 | 199.825 | 1039.429 | 1317 |
| L−E′ `p10_lgb_ne_s43` (control) | 317.5498 | 254.295 | 221.5605 | 200.027 | 1015.354 | 1439 |
| **C−E** `p10_cb_ne` | **328.2485** | **260.052** | 221.2834 | 201.430 | 978.074 | **5000** |
| L+E `p10_lgb_e` | 314.8437 | 252.836 | 220.3492 | 199.316 | 1001.615 | 898 |
| **C+E** `p10_cb_e` | **330.2925** | **256.868** | 219.2237 | 198.540 | 991.270 | **4574** |

## 2. The pre-registered rules, applied

Seed control (LightGBM, lane A): **0.223** on `jul_fwd`, **0.202** on `forward`.

**E1 — no encodings, the P2-invariant comparison.** CatBoost − LightGBM, lane A:

| fold | diff | > 3× control? |
|---|---:|---|
| `jul_fwd` | **+5.980** (CatBoost worse) | yes (0.67) |
| `forward` | **+1.605** (CatBoost worse) | yes (0.61) |

Same sign on both folds. Smaller/larger = **0.27**, just under the pre-registered 1/3, so by the letter of the
rule this is a one-fold lead rather than a result — **but the lead points against CatBoost.** There is no
reading of E1 in which CatBoost wins lane A.

**E2 — with encodings.** CatBoost − LightGBM, lane A: `jul_fwd` **+4.032**, `forward` **−0.776**. Opposite
signs; fails the shrinkage rule.

## 3. Verdict: DO-NOT-SHIP as run — and ON HOLD, not closed

**Status set by the user on 2026-09-29: on hold.** The measurement below rejects *this configuration* of
CatBoost (untuned, `learning_rate` 0.1, depth 6, `max_ctr_complexity` 1, capped on `forward`). It does not
reject the family. The open items in §4 — the pre-registered `learning_rate` 0.2 re-run for the capped
`forward` arms, the `max_ctr_complexity` 4 arm, and tuning — are the ones to pick up when it resumes. The code
path (`model_family`, default `"lightgbm"`) stays in the tree for that.

- **On the one fold where CatBoost converged, it is clearly worse.** `jul_fwd` early-stopped at 1,267 rounds
  (C−E) and 978 (C+E); lane A is +5.98 and +4.03 worse, full-fold **+11.9** and **+15.4** worse.
- **The forward fold's full-fold gains (−1.12, −1.13) come from lane B**, which the pre-registration excludes
  from the decision: CatBoost's lane B is 978 / 991 against LightGBM's 1039 / 1002. Lane B is ~6,900 rows with
  a Kish effective sample size of 7.58 (`model-b-ceiling.md`); a difference there is not measurable.
- The PRC 2024 ATOW result (CatBoost −11.6 % over LightGBM) **did not transfer to this problem.**

## 4. Caveats, recorded rather than smoothed over

- **CatBoost on `forward` hit the round cap** — 5,000 rounds (C−E) and 4,574 (C+E) against LightGBM's
  1,317 / 898 — so on that fold it was still improving and is under-trained at `learning_rate` 0.1. The
  pre-registration names this case and prescribes a single re-run at 0.2; that re-run was not reached. This
  handicaps CatBoost on `forward` only. It does **not** explain `jul_fwd`, where CatBoost stopped early on its
  own and was still 6 s worse on lane A.
- **Untuned against a LightGBM that has been adjusted over a month of work.** The ATOW comparison was
  Optuna-tuned on both sides. A tuned CatBoost could do better; nothing here measures that, and tuning on
  these folds would be selection on the evaluation set.
- `p10_cb_ne_ctr4` (the library-default `max_ctr_complexity` 4) was configured and not run.
- `p10_cb_ne_s43_jf`: landed, see §6.

## 5. What was built

`src/prc/train.py` and `src/prc/pipeline.py` gain a model-family switch whose default is `"lightgbm"`, the
only value any archived recipe uses; no `main_*` config sets it, so production is unchanged. Tests in
`tests/test_catboost_family.py`; mutation audit in `scripts/p10_mutation_audit.py`. Suite **166 passed** after the last arm
finished (15 min 09 s).

## 6. CatBoost's own seed control (`p10_cb_ne_s43_jf`), and why it strengthens the verdict

`jul_fwd`, no encodings, lane A 258.613, full 340.6153, 1,663 rounds (early-stopped, not capped).

| | lane A seed draw | full-fold seed draw |
|---|---:|---:|
| CatBoost (s42 vs s43) | **1.439** | **12.367** |
| LightGBM (s42 vs s43) | 0.223 | 1.187 |

**CatBoost is 6× noisier than LightGBM on lane A and 10× on the full fold**, on the same information.

The pre-registration measured the family gap against LightGBM's seed draw. Re-read against CatBoost's own,
which is the harder test:

| pairing (`jul_fwd`, lane A) | CatBoost − LightGBM |
|---|---:|
| CB s42 vs LGB s42 | +5.980 |
| CB s42 vs LGB s43 | +5.757 |
| CB s43 vs LGB s42 | +4.541 |
| CB s43 vs LGB s43 | +4.318 |

**All four pairings put CatBoost 4.3–6.0 s worse on lane A**, and the smallest of them is 3.0× CatBoost's own
lane-A seed draw. The lane-A verdict survives the stricter control.

The **full-fold** gap does not: +11.9 s against a CatBoost seed draw of 12.4 s. On full-fold RMSE alone,
CatBoost versus LightGBM is not distinguishable from CatBoost reshuffling its own seed. That is a second,
independent reason not to ship it — a model whose score moves 12 s between seeds is not one to spend a slot on.
