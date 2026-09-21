# P3 — The out-of-fold file and the validation protocol

Plan written 2026-09-21. Scope: `src/prc/cv.py`, `src/prc/train.py`, `src/prc/compare.py`.
**No code was modified in producing this plan.**

Marking convention, as in `../plan.md`: **[S]** = read from a source (file:line or URL given);
**[X]** = my own inference. A third mark is used for numbers I computed from this repository's own
files for this plan: **[M]** = measured here, with the recipe given in §3.0 so anyone can reproduce it.

Standing constraint that shapes the whole sequence: **`compare.py`'s `verdict()` is byte-for-byte
unchanged and must stay separable.** §4 labels every stage `REPORT` or `DECIDE`; all `DECIDE` work is
collected into one reviewable commit (§6) that touches nothing else.

---

## 1. What exists today, verified

### 1.1 `src/prc/cv.py` — fold construction

**[S]** `month_block_folds(months_present)` (`src/prc/cv.py:26-51`) builds folds over `year*100+month`
keys (`year_month_key`, `cv.py:20-23`). For the 2025 training year it emits, in this order:

| order | fold | `valid_months` | `train_months` | horizon | built at |
|---|---|---|---|---|---|
| 1 | `jan` | 202501 | 202502…202512 (11) | **−11 (backcast)** | `cv.py:38` |
| 2 | `jul` | 202507 | 202501…06, 08…12 (11) | **−5 (backcast)** | `cv.py:38` |
| 3 | `jul_fwd` | 202507 | 202501…202506 (6) | +1 (forward) | `cv.py:43-45` |
| 4 | `forward` | 202511, 202512 | 202501…202510 (10) | +1…+2 (forward) | `cv.py:46-48` |

**[S]** The `Fold` dataclass carries `name`, `valid_months`, `train_months` (`cv.py:13-17`) — so the
information needed for a fold label **already exists in memory** and is simply not written out.

**[S]** `jan` has no forward twin by construction: `before = tuple(x for x in ms if x < min(sel))`
(`cv.py:43`) is empty for January in a single-year dataset. The docstring says so at `cv.py:39-42`.

**[S]** The metric helpers: `rmse` (`cv.py:54`), `metrics` (`cv.py:58-84`, which already computes
`top20_share_of_sse` at `cv.py:74-75`), `paired_bootstrap_rmse_diff` (`cv.py:87-117`, **row**
resampling at `cv.py:107`), `per_airport` (`cv.py:120-125`).

### 1.2 `src/prc/train.py` — OOF assembly

**[S]** `run_cv` (`train.py:463-535`):

- `folds` comes from `month_block_folds` and is then filtered by `cfg.folds` if set (`train.py:480-485`).
- `oof` is initialised as one row per training row, all null (`train.py:486`).
- Per fold: fit and predict (`train.py:490-493`), record fold metrics including `by_airport`
  (`train.py:494-500`), then **`pl.coalesce([oof, _p])`** (`train.py:508-510`).
- `n_into_overall` and the `folds_pooled` / `folds_shadowed` lists record which folds actually reached
  the pool (`train.py:511-512`, `:516-517`). This was added deliberately; the docstring at
  `train.py:501-507` states that `jul_fwd`'s 190,713 predictions are silently discarded by the
  coalesce and that "the pooling is kept as it was — the logged numbers stay comparable".
- The written artefact is `oof.write_parquet(out_dir / f"oof_{rec['exp_id']}.parquet")`
  (`train.py:534`), i.e. **two columns only**.

**[S]** Verified on disk: `experiments/oof_main-f9d57a661f.parquet` has columns
`['MVT_ID_mvt', 'oof']`, 2,085,047 rows, **672,428 non-null**;
`experiments/oof_main-48c19bc830.parquet` the same shape and count;
`experiments/oof_feat_base-b9e21a829d.parquet` **328,009 non-null**.

**[S]** The experiment log confirms the coverage split is systematic, not incidental. Over the 43
records in `experiments/experiments.jsonl`:

| fold set logged | records | `overall.n` |
|---|---|---|
| `{forward}` | 24 | 328,009 |
| `{jan, jul, forward}` | 10 | 672,428 |
| `{jan, jul, jul_fwd, forward}` | 6 | 672,428 (jul_fwd shadowed) |
| `{jan}` | 2 | 153,706 |
| `{jul, jul_fwd}` | 1 | 190,713 |

**[X]** So more than half the archive (24/43) is forward-only and cannot be compared row-for-row
against the 16 pooled records at all; an inner join between the two families silently reduces to the
`forward` block, which is the slot-11 finding (`../results/11-ensembling-patterns.md` §5.3) reproduced
from the log side.

**[S]** `configs/main.yaml` does not set `folds:`, so production runs all four folds and shadows
`jul_fwd`. `n_seeds: 3` there; the `TrainConfig` default is 1 (`train.py:116`).
`train_seconds` across the archive ranges **257.2 → 2,757.6 s** (4.3–46 min per run).

### 1.3 `src/prc/compare.py` — the comparison and the verdict

**[S]** `load_pair` (`compare.py:324-332`) reads `feat_train.parquet` for
`[MVT_ID_mvt, LABEL, PROXY, MVT_TIME_UTC_mvt, ADEP_mvt]`, then **inner-joins each OOF file** and drops
rows where either prediction is null. There is no check that the two files were produced under the
same fold list; the intersection is taken silently.

**[S]** `robustness_curve` (`compare.py:55-82`) evaluates at `DROP_LEVELS = (0, 20, 200, 2000)`
(`compare.py:43`), removing the K rows with the largest `max(se_a, se_b)` (`drop_order`,
`compare.py:47-52`) and calling the **row** bootstrap at each level.

**[S] What the drop-200 arm computes, exactly.** At `K = SIGN_CHECK_LEVEL = 200` (`compare.py:44`):
mask out the 200 rows with the largest `max((y−p_a)², (y−p_b)²)`; recompute `rmse_a`, `rmse_b`,
the paired difference `RMSE(B) − RMSE(A)` and a 95% **row**-resampled CI on that difference over the
remaining rows (`compare.py:68-81` → `cv.py:87-117`). It is deliberately biased against finding an
effect — the removed rows are where the two models disagree most (`compare.py:18-19`).

**[S] What `verdict()` does, exactly** (`compare.py:112-140`), reading only the curve:

1. No `dropped == 0` entry → `(False, "no full-data comparison")`.
2. `full.ci95[0] > 0` → reject, "B is worse on all rows".
3. `full.ci95[1] >= 0` → reject, "no difference on all rows".
4. No `dropped == 200` entry → **accept**, "too few rows to run the robustness check".
5. `robust.ci95[1] >= 0` → reject, "tail-only".
6. Otherwise accept.

So the accept condition is exactly: **`full.ci95[1] < 0` AND (`robust` absent OR `robust.ci95[1] < 0`)**.
Nothing else in the module reaches it — `month_bootstrap_rmse_diff` (`compare.py:85-109`),
`group_diff` (`:143`), `per_group_diff` (`:179`), `tail_report` (`:211`) and `format_tail_report`
(`:272`) are all reporting; `main` calls `verdict(curve)` with the comment "unchanged: the curve is
still the only input to adopt/reject" (`compare.py:389`).

**[S] The "byte-for-byte unchanged" claim, verified against git.** `compare.py` has four commits:
`d9f1b45` (introduced the rule), `ea04ec9`, `5f52eb4`, `0db8e9d` (HEAD, "report the rows the decision
rule discards"). Diffing the `verdict()` body: `5f52eb4` → `0db8e9d` is **identical**, confirming the
previous addition was reporting-only. `ea04ec9` → `5f52eb4` changed only message strings: it split the
single `full["ci95"][1] >= 0` reject into a "B is worse" branch plus a "no difference" branch. Since
`ci95[0] > 0` implies `ci95[1] > 0`, **the accept/reject boolean has been unchanged since `d9f1b45`.**
That invariant is what this plan preserves.

**[S]** `month_bootstrap_rmse_diff` resamples whole `year*100+month` keys (`compare.py:85-109`) and
returns `nan` for fewer than two months (`:95`). On the pooled OOF it resamples **4 units**.

---

## 2. What is actually broken — and what is not

**[S]** sklearn is authoritative that pooling predictions from distinct models is not a generalisation
estimate: `cross_val_predict` "simply returns the labels (or probabilities) from several distinct
models undistinguished. Thus, `cross_val_predict` **is not an appropriate measure of generalization
error**"; it is appropriate for "Model blending"; and "Only cross-validation strategies that assign
all elements to a test set **exactly once** can be used"
(<https://scikit-learn.org/stable/modules/cross_validation.html>, quoted in `../results/15-postprocess-overlays.md` §3).

**[S]** H2O states the stacking precondition: base models "must be cross-validated using the same
number of folds", identical folds via `fold_assignment = "Modulo"` or a shared `fold_column`, and the
CV predictions must be preserved
(<https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html>, quoted in
`../results/11-ensembling-patterns.md` §5.1).

**[X]** Applying that to us, and separating what is genuinely wrong from what is normal:

| # | Property of our OOF | Verdict |
|---|---|---|
| 1 | A single OOF vector is stitched from several fitted models | **Normal.** Every k-fold OOF is. sklearn, H2O and jubilant-vase's `train_blend.py` all do this (`../results/11-…` §5.2). The invariant that matters — no row is predicted by a model trained on it — holds: `train.py:491-492` selects disjoint month sets. |
| 2 | `jul` and `jul_fwd` both validate 202507 and `coalesce` picks by list order (`train.py:508-510`) | **Broken.** Violates sklearn's "exactly once" precondition. Today it is benign-by-accident (`jul` always precedes `jul_fwd`, so the pool is deterministic) but it is order-dependent and the file cannot say which model won. |
| 3 | Four folds train on 6, 10, 11 and 11 months | **Broken for any fitted second stage.** Four blocks, four capacities, four horizons. A weight, a bias curve or a cap fitted on the pool is fitted across incommensurable columns (`../results/15-…` §3). |
| 4 | No fold label in the schema | **Broken, and it is what makes 2 and 3 undetectable.** The information exists (`cv.py:13-17`) and is discarded at `train.py:534`. |
| 5 | Coverage differs across experiments (672,428 vs 328,009) | **Broken at the comparison layer**, not the file layer: `load_pair` (`compare.py:324-332`) inner-joins without a guard, so a cross-family comparison silently becomes a forward-fold-only comparison. |
| 6 | `jan` and `jul` train on the future of what they validate | **Real, and partly unfixable.** The ranking set is strictly after all training data. `jul_fwd` is the honest twin; `jan` cannot have one inside 2025. |

**[X] The one thing that is *not* broken and should not be "fixed":** the pooled `overall` number. It
is a consistent, comparable series across 43 records and it is what the CV↔LB correspondence is tracked
against. Every change below is additive to it.

---

## 3. Measurements taken for this plan

### 3.0 Recipe

All numbers below come from `data/work/feat_train.parquet` joined to the named
`experiments/oof_*.parquet` files exactly as `load_pair` does (`compare.py:324-332`), then:

- `se_a = (y − p_a)²`, `se_b = (y − p_b)²`, `diff = se_b − se_a`.
- Ladder threshold (Blum & Hardt, arXiv:1502.04585): `s / √n` with `s = std(diff, ddof=1)` and `n` the
  row count. Reported in MSE units (the loss scale the rule is stated on) and converted to seconds of
  RMSE as `√(mean(se_a) + s/√n) − √(mean(se_a))`.
- Bootstraps: 300 resamples, `default_rng(0)`; row = `rng.integers(0, n, n)`; date-block = resample
  calendar dates of `MVT_TIME_UTC_mvt` with replacement and concatenate their row indices; month-block
  = the same over `year*100+month`, i.e. what `compare.py:85-109` already does.
- Row-share of net advantage: `−diff[k] / −Σ diff` for the single most favourable row, and for the top 10.

### 3.1 The Ladder threshold is enormous at K=0 and tight at K=200 — [M]

| pair | n | diff (s) | Ladder thr, K=0 | Ladder thr, K=200 |
|---|---|---|---|---|
| `det_a-cfc9417c52` → `det_b-fa2cba5913` (determinism pair) | 672,428 | +0.023 | 30.1 MSE = **+0.045 s** | 29.2 MSE = +0.064 s |
| `main-f9d57a661f` → `main-48c19bc830` | 672,428 | +34.411 | 6,372 MSE = **+10.43 s** | 130.0 MSE = **+0.286 s** |
| `feat_base-b9e21a829d` → `inv_baseline_forward-9271caca22` | 328,009 | −0.819 | 541.6 MSE = **+0.947 s** | 40.1 MSE = **+0.101 s** |
| `det_a-cfc9417c52` → `ab_split-ca87cb5af7` | 672,428 | −34.579 | 6,245 MSE = **+9.16 s** | — |

**[X] Reading.** The Ladder threshold is not a constant; `s` is the SD of the per-row squared-error
difference, so the gate automatically tightens when two models agree everywhere and blows up when they
disagree on a handful of monsters. That is the property we want, and it is a second, independent
formalisation of the same suspicion the drop-200 arm encodes. Two consequences:

- **At K=0 the Ladder gate is far stricter than anything we run today.** A ~9–10 s threshold on a pair
  whose tail moves means our full-data arm has a genuine resolution of about 9 s, not 0.25 s. Adopting
  the Ladder at K=0 as a veto would reject essentially every change this project has ever adopted
  (the weather change was −1.06 s at 200 dropped, per `configs/main.yaml`).
- **At K=200 it collapses to 0.06–0.29 s**, i.e. it lands almost exactly on our informally-asserted
  0.25 s screening resolution (`../results/17-congestion-features.md:532`). That is the first
  *derivation* of that number we have rather than an assertion.

### 3.2 Date-block bootstraps do **not** reproduce merry-quicksand's width ratio on our pairs — [M]

**[S]** merry-quicksand measured day-block / row width ratios of 1.59×, **4.60×**, 1.58×, 1.47×, 1.02×
and made date blocks mandatory for any matched or tail-targeted arm
(`../results/21-repo-merry-quicksand.md` §5 item 6).

**[M]** On our pairs (300 resamples each, CI95 width in seconds):

| pair | row | date-block (days) | month (units) | day/row |
|---|---|---|---|---|
| `det_a` → `det_b` | 0.177 | 0.180 (123) | — | **1.02×** |
| `main-f9d57a661f` → `main-48c19bc830` | 32.80 | 38.55 (123) | — | **1.18×** |
| `feat_base` → `inv_baseline_forward` | 3.265 | 3.221 (61) | — | **0.99×** |
| `det_a` → `ab_split` | 34.86 | 32.89 (123) | 22.71 (4) | **0.94×** |

**[X] Reading.** Our row bootstrap is not 1.5–4.6× too narrow; on these four pairs it is within ±20% of
the date-block bootstrap, and on two of them the date-block interval is *narrower*. The mechanism is
visible in §3.1: our per-row loss differences are so tail-dominated that the row bootstrap's variance is
already driven by whether the monster rows are drawn, which is close to what a day block does when each
monster sits on its own day. merry-quicksand's 4.60× came from a tail-targeted arm whose gain lived on a
few tail *days*; our tail lives on a few tail *rows*. **This changes the priority of the date-block
stage from high to low, and on evidence rather than by transplanting their number.** The clearly
defensible change is the *month* bootstrap (4 resampling units, `compare.py:85-109`) → date blocks
(123 units), which is a statistical-validity fix, not a width fix.

### 3.3 The Jan+Jul restriction matters; the mass weighting inside it does not — [M]

**[S]** The scored set is Jan 2026 152,719 DEP + Jul 2026 192,122 DEP = 344,841, i.e. **44.3% / 55.7%**
(`../SUMMARY.md` §A, five independent sources; `../results/24-repo-tail-sweep.md` §1).

**[M]** Our pooled OOF's Jan and Jul blocks are 153,706 + 190,713 = **344,419 rows**, a Jan share of
**44.63%** — within 0.33 pp of the scored split, and within 0.12% of the scored row count.

**[M]** For `det_a-cfc9417c52` → `ab_split-ca87cb5af7`:

| view | A | B | diff |
|---|---|---|---|
| 202501 only (n=153,706) | 376.66 | 360.50 | −16.17 |
| 202507 only (n=190,713) | 375.06 | 335.92 | −39.14 |
| Jan+Jul **row-pooled** (n=344,419) | 375.78 | 347.10 | **−28.67** |
| Jan+Jul **mass-weighted 44.3/55.7** | 375.77 | 347.02 | **−28.75** |
| Jan+Jul **even 50/50** | 375.86 | 348.43 | −27.44 |
| all-4-month pooled (what we report today) | 336.21 | 301.63 | **−34.58** |

And for `main-f9d57a661f` → `main-48c19bc830`: Jan +13.40, Jul +43.64, row-pooled +29.93,
mass-weighted +30.04, even 50/50 +28.31, all-4-month +34.41.

**[X] Three readings, and one correction to the brief.**

1. **Row-pooling the Jan+Jul OOF block *is already* the mass weighting**, to within 0.1 s. The explicit
   44.3/55.7 reweighting is a rounding correction, not a lever. The 11-point figure
   (`../results/24-…` §2.2, sqcm9896's `eda/CHALLENGE.md:44`, "50/50 flatters by ~11 pts") is about
   their holdout, not ours: on our pairs even-vs-mass is **+0.09 s** and **+1.42 s**. We should
   still weight by mass — it is free and correct — but we should not expect it to move anything.
2. **The content of this arm is the calendar restriction, and it is worth 5–6 s.** The four-month pool
   over-states the first pair's gain by 5.8 s (−34.58 vs −28.75) and the second pair's regression by
   4.4 s, because November and December are much easier months (A: 294.7 and 283.0, against 376.7 and
   375.1 for Jan and Jul) and carry 47.5% of the pooled rows while the scored set contains none of them.
   Our reported CV level is not the level of the thing being scored.
3. **The seasonally-matched arm is built from the two most anti-causal folds.** Jan comes only from the
   `jan` backcast (horizon −11) and Jul from `jul` (horizon −5) unless `jul_fwd` is made reachable.
   This is the sharpest argument for the fold label: without it, the one arm that matches the scored
   calendar is silently the one arm with no forward content.

### 3.4 The single-row-share guard bites, and is not redundant — [M]

**[S]** jubilant-vase's third criterion, enforced in code: a candidate is rejected if more than 50% of
its net advantage comes from a single row (`select_airport_membership.py:64`, `maximum_row_share=50.0`,
computed at `:136-138`) — `../results/09-repo-sergiuv11.md:196`, `../results/12-per-group-specialists.md:537`.
Their other two criteria are ≥1.0 s per-airport improvement and ≥80% of 2,000 paired bootstrap wins.

**[M]** On our pairs:

| pair | top-1 row's share of net advantage | top-10 share | 50% guard |
|---|---|---|---|
| `det_a` → `ab_split` (−34.58 s) | **13.6%** | 80.3% | **passes** |
| `feat_base` → `inv_baseline_forward` (−0.82 s) | **80.4%** | 204.2% | **rejects** |

**[X] Reading.** The guard is not redundant with drop-200, but the two disagree in both directions.
`det_a → ab_split` is the canonical tail-lottery pair that drop-200 rejects (−34.58 → −0.27) and the
single-row guard **passes** it, because its advantage is spread over ~10 monsters rather than one. The
`feat_base` pair is the reverse: the guard rejects it on one row supplying 80.4% of the net advantage
(and a top-10 share above 100%, meaning other rows move the other way), while today it is rejected only
incidentally because its full CI [−2.50, +0.76] spans zero. **[X]** Therefore the guard should be
generalised to report top-1 *and* top-10 share; a top-1-only rule as jubilant-vase wrote it would have
cleared our worst known tail lottery. Our own recorded figure — 10 OOF rows carrying 24.1% of squared
error (`docs/handoff.md:64`, `:310`) — is the reason.

### 3.5 A mechanical hazard for the fold-label change — [M]

**[M]** Polars 1.44.2 does not error when both joined frames carry a `fold` column; the second is
silently renamed `fold_right`. Reproduced: joining two frames each with `id, oof, fold` through
`load_pair`'s pattern yields `['id', 'p_a', 'fold', 'p_b', 'fold_right']`. **[X]** So adding a `fold`
column to the existing wide OOF file would not break `compare.py` loudly — it would break it quietly,
which is worse. §4 S1 takes the sidecar design instead.

---

## 4. The staged plan

Every stage is labelled **REPORT** (cannot change any adopt/reject outcome) or **DECIDE** (can).
All `DECIDE` work is deferred into the single commit D1 (§6).

### S1 — Emit a fold label. **REPORT.** Prerequisite for everything else.

**What.** In `run_cv` (`train.py:490-512`), accumulate a long frame of
`(MVT_ID_mvt, fold, pred, valid_months, train_months, n_train_months, horizon_months)` — one row per
fold-prediction, **including the shadowed `jul_fwd` rows** — and write it beside the existing file as
`experiments/oof_<exp_id>_folds.parquet`. Add to the JSONL record, per fold: `train_months`,
`valid_months`, `n_train_months` and `horizon_months` (signed: negative = backcast).

**Why a sidecar rather than a new column.** §3.5: a `fold` column on the wide file is silently
duplicated into `fold_right` by `load_pair`'s double join. A sidecar leaves
`experiments/oof_*.parquet` **byte-identical**, so all 43 existing records stay valid and nothing
downstream has to change on the same commit.

**Why long rather than wide.** It is the only shape that preserves `jul_fwd`'s 190,713 predictions,
which the coalesce at `train.py:508-510` currently discards (`train.py:501-507` documents the loss:
`overall.n` 672,428 against a fold sum of 863,141). Those rows are the *only* forward, seasonally-Jul
evidence we have, and P1/P6/P7/P10 all want them.

**Explicitly unchanged.** `oof`, the coalesce order, `overall`, `folds_pooled`/`folds_shadowed`,
`verdict()`.

**Verification.** Re-run one config (`configs/feat_base.yaml`, ~4–6 min) and assert: (a) the wide
parquet is byte-identical to the archived one for the same `exp_id`; (b) `overall.rmse_full` matches
the logged value to 4 dp; (c) the sidecar's per-fold row counts are 153,706 / 190,713 / 190,713 /
328,009 and sum to 863,141 on a four-fold run; (d) for every row present in both, the sidecar's
first-fold-in-order prediction equals the wide `oof` value.

**Effort.** ~40 lines in `train.py`, ~2 h including the verification run.

### S2 — Make comparability explicit, not uniform. **REPORT (S2a) / DECIDE (S2b).**

**[X] Recommendation: do not change `month_block_folds`.** Changing fold definitions would invalidate
all 43 records at once (§7) and buy nothing that the label does not buy. What is needed is not uniform
folds but *declared and enforced* folds.

- **S2a — REPORT.** Teach `load_pair` (`compare.py:324-332`) to read both sidecars, and print, above
  the curve: each side's fold set, per-fold non-null counts, the intersection actually compared, and a
  loud `FOLD SET MISMATCH` line when they differ. It still compares; it just stops being silent. This
  is the fix for §2 item 5 and for the 24-vs-16 archive split.
- **S2b — DECIDE.** Refuse the comparison outright (non-zero exit) when the fold sets differ, unless
  `--allow-fold-mismatch` is passed. This can turn a present-day ADOPT into "not comparable", so it is
  a decision change and belongs in D1.

**Verification.** Run S2a on one same-family pair (`main-f9d57a661f` vs `main-48c19bc830`) and one
cross-family pair (`main-f9d57a661f` vs `feat_base-b9e21a829d`); the second must print the mismatch and
report the compared set as 328,009 rows.

**Effort.** S2a ~30 lines, 1 h. S2b ~10 lines, 0.5 h.

### S3 — Ladder gate. **REPORT (S3a) / DECIDE (S3b).**

**[S]** Blum & Hardt: accept a claimed improvement only when it beats the incumbent by more than
`s/√n`, `s` = "the sample standard deviation of loss differences"
(<https://ar5iv.labs.arxiv.org/html/1502.04585>; `../results/14-validation-design.md` §7.1).

- **S3a — REPORT.** Add `s`, `n`, `s/√n` (in MSE **and** converted to seconds of RMSE at A's level) to
  every row of the robustness curve. All inputs are already computed inside
  `paired_bootstrap_rmse_diff` (`cv.py:103`); this is ~15 lines and no new computation.
- **S3b — DECIDE.** Wire it into `verdict()` as a third conjunct. **[X] Wire it at K=200 only, and
  report K=0 as context.** §3.1 measures the K=0 threshold at 9–10 s on tail-moving pairs, which would
  reject every change we have ever adopted; at K=200 it is 0.06–0.29 s and is a defensible published
  replacement for our asserted 0.25 s floor. Adopting it at K=0 is a strategy change (it says "only
  changes worth ~10 s are measurable"), and if we want that, it must be argued on its own, not smuggled
  in with the Ladder.

**Verification.** Recompute `verdict()` under S3b over the archive of comparisons and list every pair
whose outcome flips. Pre-register the expectation: `det_a → ab_split` stays REJECT (drop-200 already
rejects it); `feat_base → inv_baseline_forward` stays REJECT; the weather change (−1.06 s at K=200
against a K=200 threshold of order 0.1–0.3 s) stays ADOPT.

**Effort.** S3a ~15 lines, 1 h. S3b ~10 lines plus the replay, 3 h.

### S4 — Date-block bootstrap. **REPORT.** Lower priority than the brief assumed.

**[M] §3.2 is the reason for the downgrade:** on four of our own pairs the day/row width ratio is
0.94–1.18×, not merry-quicksand's 1.5–4.6×. So:

- **S4a — REPORT, do it.** Replace the **month** bootstrap (`compare.py:85-109`, 4 resampling units on
  the pooled OOF, 2 on a forward-only file, `nan` below 2 by `compare.py:95`) with a date-block
  bootstrap over calendar dates (123 units pooled, 61 forward-only). This is a statistical-validity fix:
  percentile CIs from 4 units are not interpretable. Keep the month line too, so the old numbers stay
  readable.
- **S4b — not recommended now.** Replacing the **row** bootstrap inside the curve — the thing
  `verdict()` reads — would be a DECIDE change for a ≤20% width move on the pairs measured. Revisit
  only if the ratio is shown to exceed ~1.5× on a tail-targeted arm (the merry-quicksand case was
  exactly such an arm); P1 and P6 will produce those arms, so re-measure then.

**Verification.** For S4a, report both month and date-block intervals on the four §3.2 pairs and
confirm the date-block interval contains the month interval's point estimate.

**Effort.** ~25 lines, 1.5 h.

### S5 — Jan / Jul / mass-weighted-pooled arm. **REPORT (S5a) / DECIDE (S5b).**

- **S5a — REPORT.** Add a block to the comparison output: `RMSE_A`, `RMSE_B`, diff and a paired CI for
  (i) 202501 alone, (ii) 202507 alone, (iii) Jan+Jul mass-weighted at 44.3/55.7, and, when the sidecar
  is present, (iv) 202507 via `jul_fwd` instead of `jul` — the forward, seasonally-matched view we
  currently throw away. Also add the mass-weighted Jan+Jul RMSE to the `overall` block of the JSONL
  record as a **new key** (`overall.rmse_janjul_massweighted`), never replacing `rmse_full`.
- **S5b — DECIDE.** Require the **mass-weighted Jan+Jul** CI to exclude zero on the improving side, in
  addition to the two existing conjuncts.

**[X] Do not adopt sqcm9896's conjunction as a veto.** Their rule ("ship only if Jan AND Jul improve",
`src/config.py:11`, `README.md:38`, `../results/24-…` §2.1) threw away a −8.6 s pooled gain
(`extreme_spec`: Jan +5.9, Jul −23.5, pooled −8.6, verdict `drop-jan-rule`, `../results/24-…` §2.4)
which is positive-expectation under the actual scoring function; and both of their documented live
failures (v6 −0.6 pooled → +7.0 live; v7 −1.2 pooled → +0.95 live) **passed** the conjunction
(`../results/24-…` §2.5). Their own post-mortems name "OVERFIT holdout monsters" and "small-slice
monster fits don't transfer" — which is what our drop-200 arm and S6 are for. Take the calendar
restriction and the mass weighting; report the per-month split as a flag; do not give either month a veto.

**Verification.** Replay the archive: report, for every past comparison, pooled diff vs mass-weighted
Jan+Jul diff, and list the pairs where the sign differs. §3.3 predicts the level will move by ~40 s and
the diff by ~5 s on tail-moving pairs; a sign flip would be a strong signal and must be inspected
individually before S5b is adopted.

**Effort.** S5a ~50 lines, 2 h. S5b ~10 lines plus the replay, 3 h.

### S6 — Single-row-share guard. **REPORT (S6a) / DECIDE (S6b).**

- **S6a — REPORT.** Report `net_advantage = −Σ(se_b − se_a)`, the top-1 row's share of it, the top-10
  share, and the ids of those rows (which `tail_report` already has the machinery for,
  `compare.py:211-257`). Report at K=0 and K=200.
- **S6b — DECIDE.** Reject when top-1 share > 50% (jubilant-vase's threshold verbatim). **[X] Consider
  a second, stricter conjunct — top-10 share > 90% — only if the replay shows it rejects a pair the
  drop-200 arm passes.** §3.4 shows top-1 alone clears `det_a → ab_split` at 13.6%; that pair's 80.3%
  top-10 share is the number that describes it.

**Verification.** Replay: report both shares for every archived comparison and cross-tabulate against
the existing verdict. S6b is worth adopting only if there is at least one pair where the guard rejects
and both existing CIs accept; if there is none, it is pure reporting and should stay REPORT.

**Effort.** S6a ~25 lines, 1 h. S6b ~10 lines plus the replay, 2 h.

### S7 — The anti-causal folds. **REPORT, plus one optional new fold.**

**[X] Do not redefine the existing folds.** The tension is structural, not a bug: the scored months are
January and July, and inside a single training year the only fold that is both forward and seasonally
matched is `jul_fwd`. `jan` cannot have a forward twin (`cv.py:43`), and the production horizon — train
on all of 2025, predict Jan and Jul 2026, +1 to +13 months out — is not reproducible by any 2025 fold at all.

What to do instead, in order:

1. **Label the horizon** (part of S1): every fold's record carries `horizon_months`, signed. `jan` = −11,
   `jul` = −5, `jul_fwd` = +1, `forward` = +1…+2. This turns "anti-causal" from a thing we know into a
   thing the artefacts say.
2. **Make `jul_fwd` reachable** (part of S1's long sidecar) and **designate it the confirm fold.** It is
   the only fold that is simultaneously forward and seasonally matched to half the scored set. This is
   the local form of the two-stage screen/confirm structure the literature converges on
   (`../results/14-…` §7.3 [S] PRC 2025 two-stage tuning against "optimization leakage"; §7.4 [S] "screen
   cheaply, confirm on a set the screen never touched").
3. **Optional new fold `dec_fwd`** — train 202501…202506, validate 202511…202512, horizon +5…+6. It is
   the closest available approximation to the production horizon and it is additive: a new fold name
   changes no existing record, because `cfg.folds` selects by name and every archived config either
   names its folds or predates this one.

**[X] What I would not do:** drop `jan` and `jul`. They carry the only January evidence in the dataset
and half the seasonal coverage, and the `cv.py:39-42` docstring is right that the seasonal difference
between months is larger than the leak — so comparing `jan` against `forward` measures season, not leak.
The leak is measurable only through the `jul` / `jul_fwd` pair, which S1 makes possible for the first
time.

**Effort.** Items 1–2 are inside S1. Item 3 is ~5 lines in `cv.py` plus one CV run (~45 min wall clock).

---

## 5. Decision-vs-report, at a glance

| stage | touches `verdict()`? | can flip an adopt/reject? |
|---|---|---|
| S1 fold sidecar + record fields | no | **no** |
| S2a mismatch warning | no | **no** |
| S2b mismatch refusal | no (new guard upstream) | **yes** — can turn ADOPT into "not comparable" |
| S3a Ladder reported | no | **no** |
| S3b Ladder as a conjunct at K=200 | **yes** | **yes** |
| S4a date blocks replace the month line | no | **no** (the month line never entered `verdict()`) |
| S4b date blocks replace the row bootstrap in the curve | **yes, indirectly** | **yes** — not recommended now |
| S5a Jan/Jul/mass-weighted reported | no | **no** |
| S5b mass-weighted Jan+Jul CI as a conjunct | **yes** | **yes** |
| S6a row-share reported | no | **no** |
| S6b row-share as a conjunct | **yes** | **yes** |
| S7 horizon labels, `jul_fwd` confirm, `dec_fwd` | no | **no** (unless the confirm stage is made mandatory, which is a D1 item) |

---

## 6. D1 — the one decision commit

**[X]** Every `DECIDE` item above lands in a single commit that changes `verdict()` and nothing else,
so that the review is about the rule and not about plumbing. Structure it so the rule is readable as one
expression:

```
adopt  ⟺  full.ci95[1] < 0
     AND  (robust200 absent OR robust200.ci95[1] < 0)      # unchanged, both existing conjuncts
     AND  robust200.diff < −ladder_threshold(robust200)     # S3b
     AND  janjul_mass.ci95[1] < 0                           # S5b
     AND  top1_share ≤ 0.50                                 # S6b
     AND  fold sets match (or --allow-fold-mismatch)        # S2b
```

Rules for D1, so that the "byte-for-byte unchanged" property remains meaningful afterwards:

1. **Nothing else in the commit.** No formatting, no renames, no new reporting. The diff should be the
   `verdict()` body, its docstring, and the plumbing that hands it the three new inputs.
2. **Keep the old rule callable.** Retain the two-conjunct rule as `verdict_v1()` and have the CLI print
   both verdicts side by side, with `v1` labelled as the rule every archived decision was made under.
   Without this, no archived decision remains interpretable (§7).
3. **Ship with the replay.** The commit message, or a `docs/` note beside it, lists every archived
   comparison whose verdict flips under D1 and why. Each new conjunct's replay (S3b, S5b, S6b) must be
   run and recorded *before* D1, as part of the corresponding REPORT stage.
4. **One conjunct at a time is acceptable and preferred if the replay is noisy.** Nothing above requires
   them to land together; the only requirement is that no `DECIDE` change rides along with a `REPORT`
   commit.

---

## 7. What is invalidated, and how to keep the old numbers readable

| change | invalidates | mitigation |
|---|---|---|
| S1 sidecar | **nothing** — the wide parquet and `overall` are untouched | verify byte-identity of the wide file on one re-run (§4 S1) |
| Adding a `fold` column to the wide file instead | would silently produce `fold_right` in `load_pair` (§3.5 [M]) | **rejected in favour of the sidecar** |
| Redefining `month_block_folds` | **all 43 records at once**: `overall.n`, `overall.rmse_full`, every per-fold number, every archived OOF parquet, and the entire CV↔LB series | **do not do it.** S7 adds a fold by name instead; a new name changes nothing existing |
| S4a date blocks | the month-bootstrap line printed in past investigation docs | keep printing the month line alongside; label it "legacy unit" |
| S4b date blocks inside the curve | every archived CI, and therefore the basis of every archived verdict | not recommended; if ever done, print both intervals and re-run the replay |
| S5a mass-weighted key in `overall` | nothing, if added as a **new key** | never rename or repurpose `rmse_full`; the CV↔LB correspondence is tracked against it |
| S5a level change in the console | reader confusion: the Jan+Jul level is ~40 s above the pooled level (§3.3: 375.8 vs 336.2) | print both, with the pooled number first and the Jan+Jul number explicitly labelled "scored-calendar subset, not comparable to the archive" |
| D1 | the *interpretation* of every archived ADOPT/REJECT: they were all made under the two-conjunct rule | keep `verdict_v1()` and print both (§6 rule 2); record in `docs/README` the commit at which the rule changed, so a reader can tell which rule any given decision was made under |

**[X] One thing that is already at risk and should be written down now.** The archive contains 24
forward-only records and 16 pooled records. Any comparison ever made across those two families was
silently a forward-fold-only comparison on 328,009 rows. S2a will reveal how many such comparisons
exist; that audit should be run and recorded even if S2b is never adopted.

---

## 8. What each stage unblocks

| downstream | needs | why |
|---|---|---|
| **P1 — unmatched-lane specialist** | **S1**, S5a, S6a/S6b, S4b re-measure | The unmatched lane is ~1.2% of rows and roughly half the squared error (`train.py:9`: unmatched 1,875.3 vs matched 218.5; `../SUMMARY.md` §E: javidmardanov 28.4% of SSE on 1.5% of rows). Every gain there lives in the tail, so the drop-200 arm rejects it by construction and the single-row guard (S6) is the only arm that can tell a structural tail win from a lottery. S1 is required because the specialist's evidence on July must come from `jul_fwd` (forward) as well as `jul` (backcast), and `jul_fwd` is currently unreachable. **[S]** merry-quicksand's Kish effective sample size on the unmatched stratum's squared errors is **22.9, not 22,219** (`../results/21-…` §5 item 7) — so any unmatched-lane CI needs the block unit re-measured (S4b), which is why S4b is deferred rather than dropped. |
| **P6 — per-airport cap** | **S1**, **S5a**, **S6b**, S2a | This is jubilant-vase's gate verbatim and it has three criteria, one of which we do not have: per-airport improvement ≥1.0 s, ≥80% bootstrap wins, and **no single row supplying >50% of the net SSE advantage** (`../results/12-…` §4, `../results/09-…:196`). Slot 12 calls criterion (iii) "mandatory for us". A per-airport rule also has to be sign-checked per fold and per month (`../results/15-…` §1.1: our own confirmed cap is 334.58 → 327.05 on **53 rows**, negative in all four OOF months and all eight affected airports) — which needs the fold label to state honestly. `per_group_diff` (`compare.py:179-203`) already computes the per-airport structure; S5a supplies the per-month axis. |
| **P7 — blended second model** | **S1**, **S2a+S2b** | This is slot 11's Step 0, stated as a hard prerequisite: "Add a `fold` column to the OOF parquet written at `train.py:534`… blocks everything else" (`../results/11-…`, Applicable §Step 0). **[S]** H2O requires identical folds across base models and preserved CV predictions; sklearn enforces one shared `cv` object structurally; jubilant-vase's `train_blend.py` runs all three libraries inside one `GroupKFold(5)` loop. Fitting NNLS weights across our current files would fit them on the 328,009-row forward block without saying so (`../results/11-…` §5.3 reason 1). S2b is what makes that impossible rather than merely visible. |
| **P10 — CatBoost as a rival** | **S1**, **S2b**, S5a, S3a | A rival-family comparison is the exact case the sklearn quotation forbids doing on a pooled heterogeneous file: if CatBoost were run with a different `cfg.folds` than LightGBM, the comparison would be decided by fold composition. S2b makes the mismatch fatal instead of silent. S5a matters because the two families may differ seasonally, and the pooled four-month level is 40 s away from the scored calendar (§3.3). S3a gives a published floor for "the two families are indistinguishable", which §3.1 suggests will be the honest answer at K=0 unless the gap exceeds several seconds. |

---

## 9. Recommended order and effort

| # | stage | kind | effort | compute | unblocks |
|---|---|---|---|---|---|
| 1 | **S1** fold sidecar + horizon fields + `jul_fwd` preserved | REPORT | 2 h | 1 verification run (~5 min) | P1, P6, P7, P10 |
| 2 | **S2a** fold-set mismatch warning + archive audit | REPORT | 1 h | none | P7, P10 |
| 3 | **S5a** Jan / Jul / mass-weighted arm + `jul_fwd` July view | REPORT | 2 h | none | P1, P6, P10 |
| 4 | **S6a** top-1 / top-10 row-share reporting | REPORT | 1 h | none | P1, P6 |
| 5 | **S3a** Ladder threshold reported at every K | REPORT | 1 h | none | P10 |
| 6 | **S4a** date blocks replace the 4-unit month bootstrap | REPORT | 1.5 h | none | — |
| 7 | **replays** for S3b, S5b, S6b, S2b over the archive | analysis | 4 h | none (OOF files exist) | D1 |
| 8 | **D1** the decision commit (S2b + S3b + S5b + S6b, `verdict_v1` retained) | **DECIDE** | 3 h | none | P1, P6 |
| 9 | **S7.3** optional `dec_fwd` fold | REPORT | 0.5 h | 1 CV run (~45 min) | P1, P10 |

**Total: ~16 h of work, one 5-minute verification run, one optional 45-minute CV run.** Steps 1–6 are
strictly additive and can be reviewed and merged individually; step 8 is the only commit that changes an
outcome.

**[X] If time is short, steps 1–3 are the load-bearing minimum.** S1 unblocks all four downstream plans;
S2a converts the most dangerous silent failure into a printed line; S5a is the only stage that measured
a multi-second correction to a number we currently believe (§3.3: the pooled level is ~40 s below the
scored-calendar level and the pooled diff over-states by ~5 s).

---

## 10. Things this plan deliberately does not do

- **[X] Does not redefine the folds.** §7 — it would invalidate the whole archive and the tension it
  would address is structural (§4 S7), not a defect.
- **[X] Does not adopt the Ladder at K=0 as a veto.** §3.1 measures that threshold at 9–10 s on
  tail-moving pairs. That may well be the true resolution of the full-data arm, but saying so is a
  strategy decision that deserves its own argument.
- **[X] Does not adopt sqcm9896's Jan-AND-Jul conjunction.** §4 S5 — it rejected a −8.6 s pooled gain
  and passed both of their documented live failures.
- **[X] Does not replace the row bootstrap with date blocks in the deciding curve.** §3.2 — the 1.5–4.6×
  ratio does not reproduce on our pairs (0.94–1.18×). Re-measure when P1 and P6 produce a
  tail-targeted arm.
- **[X] Does not fit anything on the OOF file.** Not a bias curve, not blend weights, not a CV→LB line.
  S1 is the prerequisite that makes such a fit *possible*; whether any of them is *advisable* belongs to
  P6/P7 and to `../results/14-…` §6, which found no source for a fitted CV→LB correction at all.
