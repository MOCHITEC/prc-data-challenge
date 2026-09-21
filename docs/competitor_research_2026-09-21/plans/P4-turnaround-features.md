# P4 — turnaround / arrival-linked features

Planner P4, written 2026-09-21. Marks: **[S]** = read directly out of a file in this repository (with
`file:line` and commit) or quoted from a slot report that carries its own `[S]`; **[X]** = my inference.
No `[X]` below is dressed up as `[S]`.

**Plan only. No code was changed, no branch was merged, nothing was trained.**

Source slots: `docs/competitor_research_2026-09-21/results/18-turnaround-features.md` (read in full),
`06-repo-openair.md`, `07-repo-phoenix-ops.md`.
Working branch: `claude/model-b-and-arrival-features` at `e404b7e`. Feature branch:
`codex/feature-engineering` at `8bdf4f9`. Merge base: `e6d885c`.
Current leaderboard: **284.3584** (v10, `experiments/submissions.jsonl`, `leaderboard_processed_at`
2026-09-17T06:17:06Z). **[S]**

---

## 0. The one-paragraph version

`stand_arrival_gap` is already written, tested and parity-checked on `codex/feature-engineering`, and it is
purely additive against our `HEAD` — `git diff HEAD 8bdf4f9 -- src/prc/features.py` is one new function, one
new helper, two joins and five names added to `CANDIDATE_COLS`, with **zero conflicting hunks**. **[S]**
Its headline number, **−3.2424 s as a feature** against −0.781 s as a post-hoc correction, is real but was
measured against an **eight-month** baseline, and the ten-month baseline the production `forward` fold
actually uses is already **−2.6888 s** better than that eight-month baseline on the *same* 328,009 rows.
**[S]** The gap-at-ten-months arm was started and never finished (`continuation_state.json`:
`"current": "gap_10m"`). **[S]** So the load-bearing unknown is not whether the feature works; it is how
much of the −3.24 survives once the baseline is the one we ship. Everything in this plan is arranged so
that the cheapest run answers that question first, under our own `verdict()` rule, before any column is
added on top.

---

## 1. What exists at `8bdf4f9`, exactly

### 1.1 The production implementation

**[S]** `git show 8bdf4f9:src/prc/features.py:294-336`, `stand_arrival_features(departures, arrivals)`:

| element | value |
|---|---|
| arrival pool | `ctx_src.filter(PHASE_mvt == "ARR")`, passed in from `candidate_features` (`:507`) |
| pool columns | `ADES_mvt`→`ADEP_mvt`, `STAND_mvt`, `BLOCK_TIME_UTC_mvt`→`_arrival_block`, `AIRCRAFT_TYPE_mvt`→`_arrival_type` |
| pool validity | `drop_nulls(["ADEP_mvt","STAND_mvt","_arrival_block"])`, then `sort("_arrival_block")` |
| query side | departures, `drop_nulls("MVT_TIME_UTC_mvt")`, `sort("MVT_TIME_UTC_mvt")` |
| join | `join_asof(left_on="MVT_TIME_UTC_mvt", right_on="_arrival_block", by=["ADEP_mvt","STAND_mvt"], strategy="backward", allow_exact_matches=False, check_sortedness=False)` |
| cap | `valid = gap <= 21600` (**6 h**) — no lower bound, `allow_exact_matches=False` supplies the strict `>0` |
| emitted | `stand_arrival_gap` = `MVT − _arrival_block`; `stand_aobt_turn` = `AOBT_3_flt − _arrival_block`; `stand_arrival_type_match` = `Int8(AIRCRAFT_TYPE_mvt == _arrival_type)` |
| null policy | outside the cap → `None` (a bare `pl.when(valid).then(...)`, no `.otherwise`), never 0 |
| row alignment | `departures.select("MVT_ID_mvt").join(result, how="left", maintain_order="left")` |

Wired in at `:505-511` as one more `.join` inside `candidate_features`, plus
`revision_context_features` at `:339-353` (`eobt_schedule_revision`, `delay_neighbor_deviation` — a
different family, carried in the same commit).

### 1.2 The opt-in gate

**[S]** Five names are prepended to `CANDIDATE_COLS` (`8bdf4f9:src/prc/features.py:277-281`). The gate
itself is unchanged from `HEAD`: `visible_numeric_columns` at `src/prc/train.py:230` computes
`hidden = set(CANDIDATE_COLS) - set(cfg.include_candidates) | set(cfg.exclude_features)`. A candidate is
therefore **computed always and shown only when a config names it** — so measuring one column costs a CV
run, not a feature rebuild, *provided `features.py` itself has not changed*. (It will have changed the
first time; `pipeline.py:67` `_features_version()` hashes the builder source, so the first run after the
merge rebuilds the cache.) **[S]**

Config surface: `git show 8bdf4f9:configs/codex_fe_gap.yaml` ends with
`include_candidates: [stand_arrival_gap]`. **[S]**

### 1.3 The test

**[S]** `git show 8bdf4f9:tests/test_features.py:115-145`,
`test_stand_arrival_gap_boundaries_and_keys` — one test, bringing the file from 7 to **8** test
functions (the commit message's "eight feature tests" is the file total, not eight new tests). It asserts:
row order preserved (`[4,1,2,3,5]`), `stand_arrival_gap == [None, 60, 21600, None, None]` — i.e. the
21,600 boundary is **inclusive**, a stand with no arrival is `None`, a null `MVT_TIME` is `None`, and a
different-airport arrival does not leak across the `by` key — and both halves of the opt-in gate
(`not in visible_numeric_columns(..., TrainConfig())`, `in` it with
`include_candidates=["stand_arrival_gap"]`).

### 1.4 Parity

**[S]** `8bdf4f9:docs/codex_fe_results/pipeline_parity.json`: `exact_equality: true`, `default_hidden:
true`, `opt_in_visible: true`, on **2,085,047** training and **344,841** ranking rows, for all five
columns (`stand_arrival_gap`, `stand_aobt_turn`, `stand_arrival_type_match`, `eobt_schedule_revision`,
`delay_neighbor_deviation`) — i.e. the production helper reproduces the experimental screen column
exactly.

Null rates, `8bdf4f9:docs/codex_fe_results/parity.json`: **[S]**

| column | train null | rank null |
|---|---|---|
| `stand_arrival_gap` | 0.1773 | 0.1827 |
| `stand_arrival_taxi` | 0.1773 | 0.1827 |
| `stand_arrival_delay` | 0.1773 | 0.1827 |
| `stand_sched_turn` | 0.1773 | 0.1827 |
| `stand_arrival_type_match` | 0.1779 | 0.1837 |
| `stand_aobt_turn` | 0.1859 | 0.1948 |

**[X]** The train/rank null gap is ~0.5 pp, far smaller than slot 18 §7's "pool-boundary truncation"
worry predicts for an 8 h window on a 2-month ranking file. Our 6 h cap plus a full-file arrival pool
makes the boundary effect small. It will grow if we widen the cap (see §5).

### 1.5 What is NOT implemented at `8bdf4f9`

**[S]** Only in the throwaway screen script `scripts/codex_fe/candidates.py:167-174`, never in
`src/prc/features.py`, and never refit alone as a feature:

- `stand_arrival_taxi` = the joined arrival's `TAXITIME_SEC_mvt`
- `stand_arrival_delay` = joined arrival's `MVT_TIME_UTC_mvt − SCHED_TIME_UTC_mvt` (**note: their
  `_arr_land` is the ARR *movement* time, i.e. touchdown, so this is ALDT − SLDT, not OpenAir's
  AIBT − SIBT** — see §4.1)
- `stand_sched_turn` = `SCHED_TIME_UTC_mvt(dep) − _arrival_block`

Not implemented anywhere in this repository, at any commit:

- the AOBT-anchored **second** as-of join (we have `stand_aobt_turn`, which reuses the MVT-anchored
  match; OpenAir's `since_arr_aobt` re-joins *as of* `AOBT_3_flt` — different matched arrival, §4.3)
- `arr_taxiin_2`, `arr_type` (categorical), `rwy_arr_taxiin`, `since_dep`
- Phoenix-Ops's 22 airport/runway trailing taxi-in statistics. We have a strict subset:
  `arr_taxi_in_mean_{30,60}m` and `arr_taxi_in_n_{30,60}m` at `src/prc/features.py:209-214`, airport
  scope only, no runway scope, no std, no 15 m window, no last-age / last-taxi. **[S]**
- any 24 h / 36 h window, any LIRF-scoped variant

### 1.6 The measurements that exist, and the one that does not

All from `8bdf4f9:docs/codex_fe_results/`. Evaluation population throughout:
**n = 328,009, 61 days, Nov+Dec 2025** — which is exactly the `forward` fold's holdout
(162,332 + 165,677 from `data_support.json`). **[S]** Negative is better.

| arm | baseline | RMSE | Δ | 95 % day CI | 95 % week CI | file |
|---|---:|---:|---:|---|---|---|
| gap-only vs 8-month baseline | 228.3237 | 225.0813 | **−3.2424** | [−4.282, −2.225] | [−4.226, −2.144] | `continuation_evaluation.json` |
| gap-only vs hash-noise arm | 228.1788 | 225.0813 | −3.0974 | [−4.279, −1.949] | [−4.194, −1.921] | same |
| 5-feature bundle vs gap-only | 225.0813 | 224.6608 | −0.4205 | [−1.261, **+0.492**] | [−1.324, +0.461] | same |
| all-63 bundle vs 5-feature | 224.6608 | 223.9907 | −0.6701 | [−2.064, **+0.711**] | [−1.974, +0.498] | same |
| **10-month baseline vs 8-month baseline** | 228.3237 | **225.6349** | **−2.6888** | [−4.923, −0.954] | [−5.121, −0.822] | same |
| **gap-only at 10 months** | — | — | **NOT MEASURED** | — | — | `continuation_state.json`: `"current": "gap_10m"` |

Fit settings, identical across every arm (`refit_*.json` `config` blocks): LightGBM
`lr 0.05, num_leaves 127, min_data_in_leaf 50, feature_fraction 1.0, bagging 0.8/1, lambda_l2 1.0`,
**`n_seeds: 1`**, **fixed 742 matched / 90 unmatched rounds, no early stopping**, `use_encodings: true`,
`unmatched_sched_reparam: [LIRF]`, `inner_holdout_months: 0`, 2 threads. Training rows 1,387,414 (Jan–Aug)
or 1,757,038 (Jan–Oct). Wall clock 235–410 s per arm. **[S]**

Per-airport, gap as a feature vs gap as a correction (slot 18 §8.1, from `confirmation.json` and
`continuation_evaluation.json`): 9 of 10 airports improve as a feature vs 7 of 10 as a correction; EHAM
+0.374 → −0.722, LEMD +0.428 → −0.263; LSZH stays positive at +0.493 with CI [−1.402, +3.460]. **[S]**

By month, gap vs 8-month baseline: Nov **−4.302**, Dec **−2.157**. **[S]** Both months improve.

Screen (correction-mode) deltas for the six stand columns, `screen.json`, October 2025,
n = 185,674, against an airport-only control at −0.5953: **[S]**

| column | Δ | 95 % CI | matched | unmatched |
|---|---:|---|---:|---:|
| `stand_arrival_gap` | −0.9908 | [−1.322, −0.651] | −0.655 | −10.364 |
| `stand_aobt_turn` | −0.8512 | [−1.410, −0.346] | −0.482 | −10.922 |
| `stand_arrival_type_match` | −0.7608 | [−1.319, −0.259] | −0.418 | −10.096 |
| `stand_sched_turn` | −0.6527 | [−0.937, −0.364] | −0.552 | −3.819 |
| `stand_arrival_taxi` | **−0.4410** | [−0.728, −0.148] | −0.265 | −5.280 |
| `stand_arrival_delay` | −0.4081 | [−0.690, −0.142] | −0.386 | −1.356 |
| *(control)* `noise_3` | −0.4278 | [−0.828, −0.056] | −0.179 | −7.067 |

**This table is the error slot 18 names.** `stand_arrival_taxi` ranks 5th of 6 here and is
indistinguishable from `noise_3`; it was dropped on that basis. The same column moved OpenAir
284.0967 → 282.0851, −2.01 s on the live leaderboard (slot 18 §4.5, `status.md:1577-1585`). **[S]**
The screen was a binned-residual correction, and §9 of slot 18 measures the correction mode at 4.2×
weaker than the feature mode on our own data. **[X]** A ranking produced by the weak mode should not be
trusted to order candidates for the strong mode.

One piece of direct evidence about `stand_arrival_taxi` *as a feature* does exist, from the all-63 refit
(`importance_all.json`, LightGBM total gain, Jan–Aug fit): **[S]**

| column | matched rank / 171 | matched gain share | unmatched rank / 156 | unmatched gain share |
|---|---:|---:|---:|---:|
| `stand_arrival_gap` | 34 | 0.269 % | **5** | **4.157 %** |
| `stand_aobt_turn` | **26** | 0.619 % | 145 | 0.000 % (0 splits) |
| `stand_arrival_delay` | 44 | 0.145 % | 25 | 0.706 % |
| `stand_sched_turn` | 45 | 0.134 % | 18 | 1.147 % |
| `stand_arrival_taxi` | 51 | 0.094 % | 28 | 0.580 % |
| `stand_arrival_type_match` | 75 | 0.042 % | 19 | 0.993 % |
| *(existing)* `arr_taxi_in_mean_60m` | 47 | 0.115 % | 33 | 0.423 % |
| *(existing)* `arr_taxi_in_mean_30m` | 52 | 0.091 % | 56 | 0.167 % |

**[X]** Two readings. (a) In *our* model, `stand_arrival_gap` is the load-bearing one — 5th of 156
columns in the unmatched regime — and `stand_arrival_taxi` sits exactly where our existing airport-level
`arr_taxi_in_mean_60m` sits, which is where a partly-redundant column would sit. (b) OpenAir's base
feature set for the Δ head does **not** contain an airport-level arrival taxi-in mean (slot 06 §2.6-2.7
lists `NUM0` as clocks + 14 surface counts + 8 enrich terms), so their −2.01 s bought information we
already hold part of. This is the single strongest reason to expect our `stand_arrival_taxi` gain to be
smaller than −2.01 s, and it is a reason to measure it rather than assume either number.

`stand_aobt_turn` taking **zero splits** in the unmatched model is not a surprise: unmatched means
`AOBT_3_flt` is missing, so the column is 100 % null there. **[S]** (`scalar_diagnosis.json` and the
`8bdf4f9` commit message make the same point about the other locked scalars: three of them are 100 %
null on unmatched confirmation rows and add +14.567 RMSE on 1,356 November unmatched rows.)

---

## 2. Bringing it onto this branch

### 2.1 What the divergence actually is

**[S]** Merge base `e6d885c`. `git diff --name-only e6d885c 8bdf4f9` and
`git diff --name-only e6d885c HEAD` have an **empty intersection** — `comm -12` returns nothing.
`HEAD` has touched only `docs/` and `experiments/` since the base; `src/`, `tests/`, `configs/` and
`Makefile` are byte-identical to the base on our side.

**[S]** `git diff src/prc/features.py <8bdf4f9 copy>` is exactly three additive hunks:
`+5` names in `CANDIDATE_COLS` (at `HEAD:276`), `+70` lines adding `stand_arrival_features` and
`revision_context_features` (before `_destination_distance` at `HEAD:289`), `+12` lines adding two joins
inside `candidate_features` (before `.select([...CANDIDATE_COLS])` at `HEAD:436`). `tests/test_features.py`
gains one function. Nothing is deleted or modified.

**[S]** `git merge-tree --write-tree HEAD 8bdf4f9` produces tree `13baca1` with **no conflict report**.

### 2.2 Recommendation: cherry-pick, not merge

**[X]** Merge and cherry-pick both apply cleanly; the difference is what else comes with them.
`git diff --stat e6d885c 8bdf4f9` is **15,715 insertions across 46 files**, of which 44 are
`docs/codex_fe_results/*.json` (≈15,000 lines of result artefacts), `scripts/codex_fe/*` (14 files) and
`configs/codex_fe_*.yaml` (3 files). Those artefacts are evidence and should reach `main` eventually, but
they are not the change under test and they make a code review of a 82-line feature patch unreadable.

Recommended sequence, for a human to run — **not run here**:

1. `git cherry-pick -n 8bdf4f9` then unstage everything except `src/prc/features.py` and
   `tests/test_features.py`; or equivalently
   `git checkout 8bdf4f9 -- src/prc/features.py tests/test_features.py`. Both give a byte-identical
   result, because those two files have not diverged.
2. A **second, separate commit** bringing `docs/codex_fe_results/`, `scripts/codex_fe/` and
   `configs/codex_fe_*.yaml` across unchanged, so the evidence for the −3.2424 lives in the tree that
   ships the feature. These touch nothing `HEAD` owns.
3. Do **not** merge the branch: the merge commit would put the two above into one changeset and would
   also carry `docs/codex_feature_report.md` and the branch's own `docs/features_log.md` edits into a
   branch whose `docs/` has moved on independently.

Cost: minutes. Risk: none identified — the patch is additive, the gate defaults to hidden, and
`configs/main.yaml` has no `include_candidates` key, so **the production config is unaffected by the
cherry-pick alone**. **[S]** (`configs/main.yaml` read in full; no `include_candidates`.)

### 2.3 The one thing the cherry-pick does change

**[S]** `src/prc/pipeline.py:67`, `_features_version()` hashes the feature-builder source into the cache
fingerprint, and `build_features` (`:74-99`) rebuilds when any fingerprint field differs. Editing
`features.py` therefore invalidates `data/work/feat_train.parquet` and `feat_rank.parquet`. The first
`make run` after the cherry-pick pays a full rebuild (the `8bdf4f9` code comment at
`src/prc/features.py:273-275` puts a rebuild at "about four minutes" and a CV run at "about five"; the
`refit_*.json` `seconds` fields are 235–410 s for a single-fold single-seed fit, consistent).

**[X]** Before trusting any number after the rebuild, re-run `tests/test_feature_cache.py` and confirm
the `feat_meta.json` fingerprint changed for the expected reason and nothing else moved. A silent cache
reuse across a features.py change would make every arm below meaningless.

---

## 3. Measurement protocol — this is the part that is not yet right

### 3.1 Our house rule, stated exactly

**[S]** `src/prc/compare.py`: `DROP_LEVELS = (0, 20, 200, 2000)`, `SIGN_CHECK_LEVEL = 200` (`:43-44`).
`verdict()` (`:112-139`) adopts B **only when both** the full-data 95 % CI and the worst-200-rows-removed
95 % CI exclude zero on the improving side; anything that passes at K=0 and straddles at K=200 is reported
as **tail-only** and rejected. The CI is a paired bootstrap; a month-level bootstrap is reported alongside
(`:85-110`).

**[S]** `docs/README.md:132`: 採否は 200 行除外後の CI で決める; `:148-163`: the decision fold is
**`forward` only** (train Jan–Oct 2025, validate Nov–Dec 2025) because it is the only fold that is both
forward-looking and trained on enough months; a screening run is `folds: [forward]` + `--skip-predict`,
about five minutes. `:110` records the case this rule was built for: model B at 255 leaves was
CV −34.82 s, **−0.06 [−0.61, +0.52] at K=200**, and **+13.20 s on the live leaderboard**.

**[S]** `docs/features_log.md` records **ten consecutive rejections** under this rule, the closest being
`planned_flight_s` at −0.21 [−0.49, **+0.08**].

### 3.2 Why the −3.2424 does not yet clear it

Four distinct reasons, in decreasing order of how much they worry me:

1. **Wrong baseline.** The −3.2424 is against a **Jan–Aug** fit. The production `forward` fold fits
   **Jan–Oct**, and that change alone is worth **−2.6888 s** on the identical 328,009 rows
   (`continuation_evaluation.json`, `baseline_10m_vs_baseline`). **[S]** The gap-at-ten-months arm was
   started and never finished. **[S]** **[X]** Part of what the gap buys on an 8-month fit may be
   information the extra two months of training supply anyway — for instance stand-level regularities the
   model can learn directly from September and October. I have no measurement either way, and I will not
   guess a number; this is the question stage A exists to answer.
2. **Post-hoc intervals.** `continuation_protocol.md` §2, frozen 2026-09-18: *"Add gap-only refit at the
   same settings. **This is exploratory: chosen after confirmation.** Report 95 % day and seven-day
   intervals **without calling it independent confirmation**."* **[S]** The interval family for the
   pre-registered arms was Bonferroni 98.75 % over four comparisons; the gap-only arm sits outside it at a
   descriptive 95 %.
3. **Different estimator settings.** The measured arm was `n_seeds: 1`, fixed 742/90 rounds, no early
   stopping, `inner_holdout_months: 0`, `feature_fraction: 1.0`. Production (`configs/main.yaml`) is
   `n_seeds: 3`, `num_rounds: 2000`, `early_stopping: 100`, `inner_holdout_months: 1`. **[S]**
   `docs/features_log.md` closes with the lesson that `feature_fraction: 0.8` alone moves the model
   ±0.5 s when a column is added, which is why the screen used 1.0 and a noise control. **[X]** With
   production's 0.8 (in `configs/main.yaml`'s `params`, inherited from the default) the same feature is a
   noisier proposition; the noise-control arm is not optional.
4. **No robustness level was reported.** `continuation_evaluation.json` carries
   `baseline_worst200_removed: −2.3593` for the gap arm — that is the *delta after removing the worst
   200 rows of the baseline*, a fixed diagnostic, **not** `compare.robustness_curve`'s K=200 with a CI.
   **[S]** It is encouraging (the effect does not vanish) but it is not the statistic `verdict()` reads.

### 3.3 The pre-registered confirmation — write this down before running anything

**Register this block verbatim, in the commit that cherry-picks the feature, before the first fit.**

- **Population.** The `forward` fold: fit on 2025-01..10, evaluate on 2025-11..12, n = 328,009,
  61 calendar days. One fold. Reason: `docs/README.md:148-163`. We do **not** add `jan`/`jul`; they leak
  future months and `docs/README.md:166-172` shows `jul`/`jul_fwd` shadow each other in the pool.
- **Arms.** Exactly three, fitted back to back, in this order:
  A0 `baseline` (`include_candidates: []`), A1 `noise` (`include_candidates: [noise_control]`),
  A2 `gap` (`include_candidates: [stand_arrival_gap]`).
  `noise_control` is already in `CANDIDATE_COLS` on `HEAD` (`src/prc/features.py:283`) and is documented
  at `:481-485` as existing precisely to price what adding *any* column costs under column subsampling.
  **[S]**
- **Seeds.** **3**, matching `configs/main.yaml`'s `n_seeds: 3`, with `seed: 42` so the per-seed seeds
  are 42/43/44 (`src/prc/train.py:351-356`). Reason: the decision must be made at production capacity,
  and `n_seeds: 1` leaves seed variance inside the effect. Cost is linear (`src/prc/train.py:115-116`),
  so three arms × three seeds ≈ 9 fits ≈ 35–60 min at the `refit_*.json` rate, plus one cache rebuild.
- **Everything else** at `configs/main.yaml` production values, changed in exactly one place:
  `folds: [forward]`, `--skip-predict`. Not `feature_fraction: 1.0` — the screen used 1.0 to *detect*,
  we are now *deciding*, and the decision must be taken in the model that ships.
- **Statistic.** `make compare A=<A0 exp_id> B=<A2 exp_id>`, i.e. `prc.compare`, reading
  `robustness_curve` at K = 0, 20, 200, 2000 with paired 95 % CIs, plus the month-level bootstrap.
- **Decision rule.** `verdict()` unchanged: **ADOPT iff the K=0 CI and the K=200 CI both exclude zero on
  the improving side.** No other statistic overrides it.
- **Noise floor, declared now.** A2 must also beat A1 (`make compare A=<A1> B=<A2>`) with a K=200 CI
  excluding zero. If A2 − A0 passes but A2 − A1 does not, the result is **"indistinguishable from adding
  a column"** and is not adopted. This is the control `docs/features_log.md`'s closing section says we
  would have been wrong without.
- **Pre-declared effect size for a pass.** I will not pre-register a magnitude, because the honest prior
  spans −3.24 (8-month measurement) down to something near zero (if the two extra training months supply
  the same information). What is pre-registered is the *sign and the robustness*: improving at K=0 **and**
  at K=200, and beating noise at K=200.
- **What counts as a failure.** Any of: (i) the K=0 CI includes zero; (ii) the K=0 CI excludes zero but
  the K=200 CI includes zero — reported as **tail-only**, which `docs/README.md:107-113` shows is the
  exact signature of the +13.20 s leaderboard regression; (iii) A2 does not beat A1 at K=200; (iv) the
  month-level bootstrap and the row-level bootstrap disagree in sign; (v) November and December disagree
  in sign (the 8-month measurement had both improving, −4.302 / −2.157, so a sign split at 10 months is
  new information and should stop the stage).
- **What we do on failure.** Stop. Do not re-cut the interval, do not switch folds, do not switch to
  `feature_fraction: 1.0` to recover the effect, do not gate the feature per airport (§6). Record the
  number in `docs/features_log.md` as rejection #11 and move the effort to P7 (capacity), which slot 18
  §0 argues is where the −7.11 actually came from.
- **One pre-declared exemption.** If A2 passes at K=0 and fails at K=200, and the K=200 point estimate
  is still improving, one — and only one — follow-up is permitted: refit A2 with `n_seeds: 5` to see
  whether the K=200 interval was seed noise. That follow-up is registered here so it is not a post-hoc
  choice, and its outcome is reported as a fifth comparison whether or not it helps.

**[X]** Why three arms and not more: every extra arm widens the multiplicity family, and the thing we
most need is a clean yes/no on one column at production settings. The rest of the catalogue in §4 is
sequenced *after* this, each as its own pre-registered A/B against the then-current best.

---

## 4. The additional columns, in priority order

Every definition below is complete enough to implement without opening a competitor repository. All of
them live in `stand_arrival_features` in `src/prc/features.py` unless stated, all are emitted as
candidates (hidden by default), all null outside their window rather than zero-filled, and all take the
arrival pool that `candidate_features` already passes:
`ctx_src.filter(pl.col("PHASE_mvt") == "ARR")` — note `ctx_src` is `mask_departures(...)`ed, so DEP
`BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` are already null and cannot leak (`src/prc/features.py:47-52`,
`:524`). **[S]**

### Priority 1 — `stand_arrival_taxi`

The joined arrival's taxi-in seconds. This is the column OpenAir's log credits with
284.0967 → 282.0851, **−2.0116 s** on the live leaderboard (v13 → v16, slot 06 line 390, `status.md:1583`).
**[S]**

| element | value |
|---|---|
| source column | ARR `TAXITIME_SEC_mvt`, aliased `_arrival_taxi` in the pool select |
| join | **reuses the existing join** — no new as-of pass |
| key | `(ADEP_mvt ← ADES_mvt, STAND_mvt)` |
| time window | `0 < gap <= 21600` (identical `valid` mask as `stand_arrival_gap`; §5 widens both together) |
| self-exclusion | structural — a DEP row is never in an ARR pool (Phoenix-Ops make the same point, slot 07 §1.5) |
| tie handling | inherited from `join_asof`: Polars backward as-of with `allow_exact_matches=False` picks **one** row among equal right keys, unspecified which. Phoenix-Ops *reject* ties at the stand level (`arrival_features.py:70-71`, NaN) and *average* them at airport/runway level (`:54-57`). **[X]** Ours silently picks one. See the risk in §7. |
| null policy | `None` when out of window or unmatched; **never 0** |
| pool validity | **add** `TAXITIME_SEC_mvt.is_between(0, 7200)` to the pool filter, matching Phoenix-Ops `arrival_features.py:24` — a negative or 3-hour taxi-in is a data error, not a long taxi. **[X]** This is a change from `8bdf4f9`, whose pool has no taxi validity filter (it does not read taxi at all). Applying it means the pool loses a few rows and `stand_arrival_gap` changes on those rows, which **breaks the `pipeline_parity.json` exact-equality guarantee**. Either apply the filter as a separate null-out on `stand_arrival_taxi` only (keeps gap byte-identical, recommended) or re-run parity. |
| dtype | Float64 like the rest of `CANDIDATE_COLS` (the pipeline casts nothing here) |

Expected effect: **[S]** −2.01 s on OpenAir's leaderboard, in a model with no airport-level arrival
taxi-in mean. **[S]** Our own screen put it at −0.441 as a correction, indistinguishable from `noise_3`
at −0.428. **[S]** In our all-63 refit it is gain-rank 51/171 matched, 28/156 unmatched, side by side
with our existing `arr_taxi_in_mean_60m`. **[X]** My best guess is that we get materially less than
−2.01 s because we already hold the airport-level version; I would be unsurprised by anything in
[−1.0, 0.0] and I would be surprised by −2.0. Measure it as its own arm against the stage-A winner.

### Priority 2 — `stand_arrival_delay` (rebuilt correctly)

Inbound lateness of the joined arrival.

**[S]** OpenAir's `arr_delay` is `aibt − SCHED_TIME` — **in-block minus scheduled in-block**
(`matched_submit.py:160`, slot 06 line 258, slot 18 §4.1). **[S]** Our screen's `stand_arrival_delay`
at `scripts/codex_fe/candidates.py:172` is `(_arr_land − _arr_sched)` where `_arr_land` is the ARR's
`MVT_TIME_UTC_mvt` — **touchdown minus scheduled**, a different quantity. **[X]** The two differ by the
arrival's own taxi-in, which is priority 1; using the in-block version makes
`arr_delay = since_arr_anchor − scheduled` decomposable against it, and the touchdown version is
collinear with `taxi + delay`. Implement the in-block version and, if cheap, keep the touchdown version
as a second column so the choice is measured rather than argued.

| element | value |
|---|---|
| definition | `_arrival_block − _arrival_sched`, seconds, where `_arrival_sched` = ARR `SCHED_TIME_UTC_mvt` |
| join / key / window / self-exclusion / ties | identical to priority 1, same as-of pass |
| null policy | `None` when the arrival has no `SCHED_TIME_UTC_mvt`, or out of window |
| sign | positive = the inbound was late |

Expected effect: **[S]** part of OpenAir's v17 bundle, 282.0851 → 281.8799, **−0.2052 s for four columns
together**. **[S]** Our screen: −0.408 as a correction, matched −0.386 / unmatched −1.356 — notably the
*least* unmatched-driven of the six, which is the opposite profile to `stand_arrival_gap`.

### Priority 3 — the true AOBT-anchored twin: `stand_arrival_gap_aobt`, `stand_arrival_taxi_aobt`

**[S]** OpenAir run a **second, independent** as-of join with `left_on = AOBT_3_flt`
(`matched_submit.py:194-200`, join B in slot 18 §4.2) and emit `since_arr_aobt` and `arr_taxiin_aobt`
from *that* match, window `0 < x < 8 h`. **[S]** Our `stand_aobt_turn` is **not** that: it is
`AOBT_3_flt − _arrival_block` computed from the **MVT-anchored** match
(`8bdf4f9:src/prc/features.py:329-331`). When an arrival on-blocks *between* this flight's push-back and
its take-off, the two disagree — and slot 07 §1.5's "availability caveat" says exactly such arrivals are
in the MVT-anchored pool.

| element | value |
|---|---|
| new join | second `join_asof`, `left_on="AOBT_3_flt"`, `right_on="_arrival_block"`, `by=["ADEP_mvt","STAND_mvt"]`, `strategy="backward"`, `allow_exact_matches=False`; the departure frame must be re-sorted by `AOBT_3_flt` and rows with null `AOBT_3_flt` dropped before the join and re-joined back as null |
| emits | `stand_arrival_gap_aobt` = `AOBT_3_flt − _arrival_block_aobt`; `stand_arrival_taxi_aobt` = that arrival's taxi-in |
| window | same cap as the MVT-anchored pair |
| null policy | `None` where `AOBT_3_flt` is null — that is **1.53 %** of ranking DEP rows and **is exactly the unmatched lane** (slot 06 §0, `status.md:102`). **[S]** So these two columns are structurally 100 % null on unmatched rows, the failure mode the `8bdf4f9` commit message names (+14.567 RMSE on 1,356 November unmatched rows for three such scalars) and that `importance_all.json` confirms (`stand_aobt_turn`: 0 splits in the unmatched model). **[X]** Ship them to the **matched** model only, via a matched-specific include list, or accept that they are dead weight in model B. |

Expected effect: **[S]** part of the same −0.2052 s v17 bundle. **[X]** Low. Worth doing mainly because
it corrects a definition we currently have wrong, not because of its size.

### Priority 4 — `stand_arrival_taxi_2`

**[S]** OpenAir's `arr_taxiin_2`: taxi-in of the arrival *before* the matched one at the same stand,
built as `arr_taxiin.shift(1).over([airport, STAND_mvt])` on the arrival pool sorted by
`(airport, STAND_mvt, aibt)`, then carried through the same as-of join
(`matched_submit.py:162-164, 227-228`).

| element | value |
|---|---|
| construction | on the arrival pool, **before** the join: `sort(["ADEP_mvt","STAND_mvt","_arrival_block"])`, then `_arrival_taxi_2 = pl.col("_arrival_taxi").shift(1).over(["ADEP_mvt","STAND_mvt"])` |
| caution | the pool must then be re-sorted by `_arrival_block` alone for `join_asof`; Polars `join_asof` requires the right frame sorted on the right key. **[X]** Getting this order wrong silently produces garbage — `check_sortedness=False` is already set in our code (`8bdf4f9:features.py:322`), which disables the guard that would catch it. Add an explicit assertion. |
| window | the *matched* arrival's window governs; no separate window on the predecessor |
| null policy | `None` for the first arrival at each stand, and wherever the primary match is null |

Expected effect: **[S]** part of OpenAir's v18, 281.8799 → 281.7196, **−0.1603 s shared with
`arr_type` and `rwy_arr_taxiin`**.

### Priority 5 — `stand_arrival_type` (categorical) and `stand_arrival_carrier_match`

**[S]** OpenAir pass the joined arrival's `AIRCRAFT_TYPE_mvt` as a **categorical** (`arr_type`,
slot 06 line 259). **[S]** Phoenix-Ops additionally emit `completed_arrival_stand_carrier_equal`: 1.0 if
the `^([A-Z]{2,4})` prefix of the departure's `FLIGHT_mvt` equals that of the matched arrival's
(`arrival_features.py:75-76`, slot 07 §1.5). We already have the type-equality flag
(`stand_arrival_type_match`) but neither the raw type nor the carrier flag.

**Implementation constraint, ours, not theirs:** `visible_numeric_columns`
(`src/prc/train.py:223-236`) gates only **numeric** columns; categoricals come from the module-level
`CATEGORICALS` list (`src/prc/features.py:33-45`) filtered by `cfg.exclude_features`
(`src/prc/train.py:374`). **[S]** So a new categorical **cannot** be shipped hidden-by-default through
`CANDIDATE_COLS`; it is visible the moment it is added to `CATEGORICALS`. **[X]** Either extend the gate
(add a `CANDIDATE_CATEGORICALS` tuple and subtract it in the same place) or hold `stand_arrival_type`
back. I recommend extending the gate once, in its own commit, before any categorical candidate — the
`8bdf4f9` docstring at `src/prc/train.py:126-129` records that v6 shipped nine unintended columns
precisely because the gate was opt-out.

`stand_arrival_carrier_match` is numeric (Int8) and needs no gate change: departure
`FLIGHT_ID_mvt`/`CALLSIGN_flt` prefix vs the arrival's. **[S]** Slot 18 §5.2 warns the same-callsign rate
at a stand is only **2 %**, so expect a very sparse flag.

**[S]** Hard stop, repeated from slot 18 §5.1: **do not use `FLIGHT_ID_mvt` as a rotation/tail key.**
315,272 IDs appear twice and 315,262 of those are the same city-pair flight seen as DEP then ARR; the
paired ARR is *after* the DEP (median Δt 5,065 s). Prefix matching on the designator is legal; identity
join on the ID is future information.

### Priority 6 — `stand_prev_dep_gap` (`since_dep`)

**[S]** OpenAir's `since_dep`: take-off minus the previous **departure's** take-off from the same stand,
window `1 s < since_dep < 12 h` (`matched_submit.py:213-225`). Slot 18 §4.3 flags that their backward
as-of join matches the row against itself at gap 0 and only the `> 1 s` guard removes it — **exclude self
explicitly**, with `allow_exact_matches=False` plus a filter, not with a magnitude guard.

| element | value |
|---|---|
| pool | DEP rows of `ctx_src` — take-off times only. **Never** their `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`: blanked in ranking, slot 18 §7 marks reading them **ILLEGAL** |
| key | `(ADEP_mvt, STAND_mvt)` |
| window | `1 < x < 43200` (12 h) |
| self-exclusion | `allow_exact_matches=False` **and** an explicit `MVT_ID_mvt != _prev_dep_id` filter |
| null policy | `None` outside the window and for the first departure at each stand |

Expected effect: **[S]** part of the −0.2052 s v17 bundle. **[S]** OpenAir's own oracle test (E71c) on
exactly this construction: `corr(T, gap_block) = 0.008` unmatched, 0.043 LIRF-unmatched — *"the oracle has
no signal"*; and E68 found extra previous-DEP stand clocks **neutral** on top of the shipped set.
**[X]** This is the lowest-expected-value item that any competitor actually shipped. Do it only if
priorities 1–5 have already paid.

### Priority 7 — Phoenix-Ops's 22 airport/runway trailing taxi-in statistics

**[S]** `arrival_features.py:36-61`, slot 07 §1.5, slot 18 §1.3. For
`label ∈ {airport: group [airport], runway: group [airport, runway]}` and `m ∈ {15, 30, 60}` minutes,
over arrivals whose **on-block** time is strictly before the departure's take-off
(`searchsorted(side='left')`):

| column | definition |
|---|---|
| `completed_arrival_{label}_{m}m_count` | n arrivals with on-block in `[q − 60m, q)` |
| `completed_arrival_{label}_{m}m_mean` | mean arrival **taxi-in seconds** in that window |
| `completed_arrival_{label}_{m}m_std` | population sd, `sqrt(max(0, E[x²] − mean²))` |
| `completed_arrival_{label}_last_age_seconds` | `q − t_prev`, only if `q − t_prev ≤ 7200`, else NaN |
| `completed_arrival_{label}_last_taxi_seconds` | taxi-in of the last arrival, **averaged over all arrivals tied at that on-block timestamp** |

2 labels × (3×3 + 2) = 22. Pool validity: `landed.notna() & blocked.notna() & blocked >= landed &
taxi ∈ [0, 7200]`. Keys normalised `strip().upper()`, empty → `UNKNOWN`. Emitted `float32`.
Counts are 0 when empty; means/stds NaN.

**What we already have** (`src/prc/features.py:209-214`): `arr_taxi_in_mean_{30,60}m` and
`arr_taxi_in_n_{30,60}m`, airport scope, `lag_sec=0`, anchored on arrival in-block. **[S]** That is
4 of the 22. **Missing: the entire runway scope (11 columns), the standard deviations (6), the 15 m
window (3), and both last-age/last-taxi pairs (4).** The runway scope is the interesting half — slot 18
§6 records `stand × runway` group means at η² ≈ 0.2525 against airport alone at η² ≈ 0.1121
(`OpenAir/status.md:144-150`).

**[X]** Implementation note: `context_features` already loops per airport and already holds `arr_inblock`
and `taxi_in` arrays (`:165`, `:210`) and already computes runway-scoped counts
(`rwy_tko_prev_15m/60m`, `:197`). Adding the runway-scoped arrival-taxi block is a nested loop over
`RUNWAY_mvt` inside the existing airport loop reusing `_window_mean`, not a new subsystem. Cost:
**[X]** maybe an hour of work and a measurable increase in feature-build time (the airport loop is
already the expensive part).

Expected effect: **[S]** these 22 plus the 4 same-stand columns together are the v7 step, −2.6224 s
officially (288.371 → 285.749), at **25 % blend weight** inside their ensemble (METHOD.md:153).
**[X]** Attributing a share of −2.62 to the 22 is not possible from their log; they never ablated the
blocks apart. Treat this as one bundle, measured as one arm.

### Not planned: two named gaps in the published work

**[X]** Slot 18 §6 names two things nobody has tried: the joined arrival's **origin** (`ADEP_mvt` of the
ARR row — available, untested by any competitor read) and **stand occupancy duration** computed directly
(rather than implied by `since_arr − since_dep`). Both are cheap given priorities 1 and 6. I am not
planning them, because our measurement budget is the scarce resource and there is no external number to
justify spending an arm on either. Recorded so they are not lost.

---

## 5. Widening the cap, as its own measurable step

Ours is **6 h** (21,600 s, `8bdf4f9:src/prc/features.py:327`). The four published values: **[S]**

| who | scope | cap | source |
|---|---|---:|---|
| us | same stand | 6 h | `8bdf4f9:features.py:327` |
| OpenAir | same stand, matched branch | 8 h | `matched_submit.py:173-181`, slot 06 line 255 |
| Phoenix-Ops | same stand | **24 h** | `arrival_features.py:66-74`, slot 07 §1.5 |
| Phoenix-Ops | airport / runway last-age | 2 h | `arrival_features.py:58-59` |
| OpenAir | same stand, **LIRF gate-hold model only** | **36 h** | `run_e74_make_v20.py:51`, slot 06 line 255 |
| OpenAir | previous same-stand departure | 12 h | `matched_submit.py:219-225` |

**[X]** These are not four opinions about one quantity. Phoenix-Ops's 24 h and OpenAir's 36 h are
attached to *different consumers* — a 123-column global CatBoost and a 383-row LIRF specialist
respectively — and OpenAir's 8 h is the only one attached to a plain global matched Δ-model, which is the
closest analogue to what we run. The widening is therefore a **feature question with a fixed answer only
per consumer**, and must be measured, not copied.

### What widening does mechanically

The 6 h cap is a `pl.when(valid)` null-out **after** the join, not a filter on the join
(`8bdf4f9:features.py:326-336`). **[S]** So widening changes exactly one constant and nulls fewer rows;
it never changes which arrival is matched. Null rate falls from 17.73 % train / 18.27 % rank toward the
structural floor (the fraction of departures whose stand has *no* prior arrival in the file). **[X]** The
newly-admitted rows are, by construction, exactly the long-gap regime where slot 18 §8.2's EHAM RASAS
mechanism says the arrival is a **different aircraft** — Schiphol's published rule tows aircraft off-gate
at turnarounds ≥ 210 min (cat ≥5) / ≥ 170 min (cat ≤4), with a *different* arrival and departure stand.
So widening is precisely the change most likely to reintroduce the EHAM regression that moving from
correction to feature fixed. That is the thing to look at in the per-airport breakdown.

### The step

- **One constant, exposed.** Lift `21600` to a module constant, e.g.
  `STAND_ARRIVAL_CAP_SEC = 21_600`, so the arm is a one-line change with an obvious diff.
  **[X]** Do *not* make it a `TrainConfig` field: it lives in the feature builder, and every value
  change invalidates the feature cache (`pipeline.py:67`) — a config knob would hide a four-minute
  rebuild behind a yaml edit and make cache-hit reasoning wrong.
- **Ladder, in one pre-registered family: 6 h (incumbent) → 8 h → 24 h.** Three arms, three cache
  rebuilds, one `forward` fold each, `n_seeds: 3`, compared pairwise against the 6 h incumbent with
  `prc.compare`. Bonferroni the two challenger comparisons to **97.5 %** and say so before running.
  Skip 36 h: its only published home is a LIRF-only specialist we do not have (that is P1 / P5
  territory, not this plan).
- **Run this step only after stage A and stage B have settled**, because a wider cap changes
  `stand_arrival_gap`, `stand_arrival_taxi` and `stand_arrival_delay` simultaneously, and there is no
  point tuning the window of a column that did not pass.
- **Mandatory diagnostic, whatever the verdict:** the per-airport delta table, specifically EHAM, LEMD
  and LSZH. **[S]** As a feature at 6 h these are −0.722, −0.263 and +0.493. If widening pushes EHAM
  positive again, that is the RASAS mechanism showing up and the wider cap should be rejected **even if
  the pooled number improves** — because the pooled number is 61 days of 2025 and the mechanism is a
  standing regulation that will still be in force in 2026. **[X]** This is a judgement call and is
  recorded as one.
- **Failure:** neither 8 h nor 24 h clears the 97.5 % K=200 bar. Keep 6 h, record it, stop. This is the
  likeliest outcome **[X]** — the cap governs 17.7 % of rows in one direction only and our 6 h figure is
  already inside the range two teams shipped.

---

## 6. What we will not do, and why

**[S]** All six are sourced; all six are cheap-looking and all six lost.

1. **No per-airport gate for EHAM / LEMD / LSZH.** Six independent citations in slot 18 §8.3:
   Phoenix-Ops v5 (airport-specific weights 323.712, worse than fixed), Phoenix-Ops v9 (per-airport
   experts shipped at 12.5 % effective weight inside a global blend, never selectively; "the airport
   model alone worsens January and February"), OpenAir E67 (240.50, unstable), E75 (237.58 vs 235.86;
   NNLS weights do not transfer, JJ 82 % vs Dec 0 %), E25 (473.61 vs 368.03), E13/E18 (every
   airport-conditional gate had precision ≈ 0 or reversed between splits). Keep `ADEP_mvt` a categorical
   and let the model decide — which our own numbers show already fixes EHAM and LEMD.
2. **No hard bound `taxi_out <= stand_arrival_gap`.** OpenAir E72: LIRF-unmatched RMSE
   3,896 → **11,579**.
3. **No post-hoc binned correction from this family.** Our own data: −0.781 as a correction vs −3.242 as
   a feature, and the correction form is what produced the three positive airports.
4. **No `FLIGHT_ID_mvt` rotation key.** §4 priority 5.
5. **No external ADS-B registration chaining.** OpenAir E77, OPDI `icao24` turnaround: 236.06 vs 235.94,
   **worse**.
6. **No reading of another ranking DEP's `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`.** Blanked;
   `mask_departures` already enforces this on the context frame and must keep doing so.

---

## 7. Risks

| # | risk | evidence | mitigation |
|---|---|---|---|
| R1 | **The −3.2424 does not survive the 10-month baseline.** | The 10-month baseline alone is −2.6888 on the same rows; the gap-at-10-months arm was never run. **[S]** | This is stage A. It is the first thing measured and the cheapest. |
| R2 | **It survives CV and loses on the leaderboard.** | `docs/README.md:107-113`: model B 255 leaves, CV −34.82, K=200 −0.06, LB **+13.20**. | The K=200 gate is exactly this defence. Additionally: this family's benefit is concentrated in the unmatched lane (screen unmatched −10.364 vs matched −0.655; `importance_all` rank 5/156 unmatched), and the unmatched lane is 1.2 % of rows and the tail that K=200 trims. **[X]** Expect the K=200 number to be much smaller than K=0 and treat a large K=0/K=200 gap as a warning, not a win. |
| R3 | **`join_asof` tie behaviour is unspecified.** Both competitors handle ties deliberately and differently: Phoenix-Ops **reject** stand-level ties (NaN) and **average** airport/runway ties; we do neither. **[S]** | slot 07 §1.5, `arrival_features.py:54-57, 70-71` | Measure how many rows have tied `(ADEP, STAND, BLOCK_TIME)` arrival timestamps before deciding. If it is a handful, note it and move on; if it is material, adopt Phoenix-Ops's reject-to-null and re-run parity. Either way, `check_sortedness=False` at `features.py:322` disables Polars' own guard — add an explicit sortedness assertion. |
| R4 | **`stand_arrival_taxi` is redundant with `arr_taxi_in_mean_{30,60}m`.** | `importance_all.json`: it sits at the same gain rank as those columns. **[S]** | It is measured as its own arm against the stage-A winner, with the noise control. If it fails, that is an answer, not a loss. |
| R5 | **Adding a categorical silently un-gates it.** `CANDIDATE_COLS` gates numerics only; v6 shipped nine unintended columns for exactly this reason. **[S]** `src/prc/train.py:126-129, 223-236` | Extend the gate (`CANDIDATE_CATEGORICALS`) in its own commit before any categorical candidate, with a test mirroring `test_stand_arrival_gap_boundaries_and_keys`'s two gate assertions. |
| R6 | **AOBT-anchored columns are 100 % null on the unmatched lane**, which is where this family pays most. | `parity.json` `stand_aobt_turn` rank-null 0.1948 vs 0.1827; `importance_all.json` 0 splits unmatched; `8bdf4f9` commit message: three such scalars add **+14.567 RMSE** on 1,356 November unmatched rows. **[S]** | Ship AOBT-anchored columns to the matched model only, or not at all. Never evaluate them pooled without the matched/unmatched split. |
| R7 | **Cache staleness.** Every change to `features.py` invalidates the feature cache; a silent reuse would make an arm meaningless. **[S]** `pipeline.py:55-99` | Check `feat_meta.json` between arms; keep `tests/test_feature_cache.py` green. |
| R8 | **Parity drift.** Adding a taxi-validity filter to the arrival pool changes `stand_arrival_gap` on a few rows and invalidates `pipeline_parity.json`'s exact-equality claim. **[S]** | Apply taxi validity as a null-out on the taxi column only, or re-run the parity script (`scripts/codex_fe/check_pipeline.py`) and update the artefact. |
| R9 | **Multiplicity creep.** §4 lists seven priorities; running all of them as free-standing arms against a moving best is a garden of forking paths. **[X]** | Stages are sequential, each with a pre-declared family and a stop rule. Priorities 4–7 are bundles, not individual columns. |

---

## 8. The staged plan

Every stage: one `forward` fold, `n_seeds: 3`, production `configs/main.yaml` settings otherwise,
`--skip-predict`, `prc.compare` against the stage's stated incumbent, `verdict()` unchanged, plus a
noise arm wherever a new column is added. "Effort" is my estimate. **[X]** on all effort figures.

### Stage 0 — land the code (no measurement)

- **Files:** `src/prc/features.py` (+82 lines), `tests/test_features.py` (+1 test) from `8bdf4f9`;
  second commit for `docs/codex_fe_results/`, `scripts/codex_fe/`, `configs/codex_fe_*.yaml`.
- **Functions:** `stand_arrival_features`, `revision_context_features`; five names into `CANDIDATE_COLS`.
- **Method:** `git checkout 8bdf4f9 -- src/prc/features.py tests/test_features.py`. **Not a merge**
  (§2.2). Conflicts: none — verified by `git merge-tree --write-tree` and by an empty
  `comm -12` of the two change sets. **[S]**
- **Verify:** `make test` (8 feature tests), `make lint`, and confirm `configs/main.yaml` still has no
  `include_candidates` so production is untouched.
- **Effort:** under an hour, most of it the first feature rebuild.
- **Failure:** tests red, or `feat_meta.json` does not change. Stop and diagnose before any fit.

### Stage A — the pre-registered confirmation of `stand_arrival_gap` *(the whole point)*

- **Arms:** A0 baseline / A1 `noise_control` / A2 `stand_arrival_gap`, as §3.3.
- **Configs:** three new yaml files under `configs/`, differing only in `include_candidates` and `name`,
  all with `folds: [forward]`.
- **Measurement:** `make compare A=A0 B=A2` and `make compare A=A1 B=A2`.
- **Expected effect:** **[S]** −3.2424 [−4.282, −2.225] at 8 months, 1 seed, `feature_fraction: 1.0`;
  Nov −4.302 / Dec −2.157; vs noise −3.0974; `baseline_worst200_removed` −2.3593. **[X]** At 10 months,
  3 seeds and `feature_fraction: 0.8` I expect materially less, and I decline to name a number.
- **Failure:** the five conditions in §3.3. On failure, stop the theme and hand the budget to P7.
- **Effort:** ~1 h of setup, ~1 h of compute (9 fits at 235–410 s, serial, 2 threads).

### Stage B — `stand_arrival_taxi` (+ `stand_arrival_delay`, in-block form)

- **Precondition:** stage A ADOPTed.
- **Files/functions:** `stand_arrival_features` gains `_arrival_taxi` and `_arrival_sched` in the pool
  select and two emitted columns; two names into `CANDIDATE_COLS`; one test extending
  `test_stand_arrival_gap_boundaries_and_keys` with a taxi value and an out-of-window null.
- **Arms:** B0 = stage-A winner; B1 = + `stand_arrival_taxi`; B2 = + both; B3 = noise control.
  Pre-declare Bonferroni **97.5 %** for the two challenger comparisons against B0.
- **Expected effect:** **[S]** OpenAir −2.0116 (v13→v16) for `arr_taxiin` + `since_arr` together, in a
  model with no airport-level arrival taxi-in mean; **[S]** our screen −0.441 (correction mode,
  indistinguishable from `noise_3` at −0.428); **[S]** gain rank 51/171 matched in our all-63 refit.
  **[X]** I expect a fraction of OpenAir's number because of R4.
- **Failure:** neither B1 nor B2 clears K=200 at 97.5 %. Record and stop; do not proceed to stage C.
- **Effort:** ~2 h of work (including the parity question in R8), ~1.5 h of compute.

### Stage C — the definitional corrections: true AOBT-anchored twin, `stand_arrival_taxi_2`

- **Precondition:** stage B ADOPTed.
- **Files/functions:** a second `join_asof` pass inside `stand_arrival_features` (priority 3) and a
  `shift(1).over(...)` on the pool (priority 4), with the explicit sortedness assertion from R3.
- **Arms:** one bundle arm plus noise, against the stage-B winner. **Report matched and unmatched
  separately and decide on the matched number**, per R6.
- **Expected effect:** **[S]** OpenAir's v17 + v18 tail, −0.2052 and −0.1603, each shared across
  several columns. **[X]** Small. The reason to do it is that `stand_aobt_turn` is currently a
  different quantity from the thing OpenAir measured, and that should be corrected before it is judged.
- **Failure:** bundle does not clear K=200. Keep the stage-B winner.
- **Effort:** ~3 h work, ~1 h compute.

### Stage D — the cap ladder

As §5: 6 h / 8 h / 24 h, Bonferroni 97.5 %, mandatory per-airport EHAM/LEMD/LSZH diagnostic with a
reject-on-EHAM-regression rule declared in advance. ~1 h work (one constant), ~2 h compute (three cache
rebuilds dominate).

### Stage E — Phoenix-Ops's runway-scoped arrival taxi-in block

- **Precondition:** at least stage B ADOPTed. Independent of C and D.
- **Files/functions:** `context_features` in `src/prc/features.py`, a runway-scoped inner loop reusing
  `_window_mean` and `_counts_in_windows` (priority 7). 18 new columns after subtracting the 4 we have.
- **Arms:** one bundle arm plus noise, against the then-current best.
- **Expected effect:** **[S]** the 22+4 bundle is Phoenix-Ops's v7 step, −2.6224 official, at 25 % blend
  weight; **[S]** stand×runway group means η² ≈ 0.2525 vs airport-alone η² ≈ 0.1121. **[X]** No
  competitor ablated the 22 from the 4, so no per-block number exists.
- **Failure:** bundle does not clear K=200. This is the arm I would most expect to fail on our data, since
  `docs/features_log.md` records four congestion/operating-mode families all rejected and
  `DISCOVERY_REPORT.md:292` puts congestion correlations at 0.05–0.17 and *"redundant given MVT−AOBT"*.
- **Effort:** ~4 h work (it touches the hot loop and will slow the feature build), ~1 h compute.

### Explicitly out of scope, handed to other planners

- **Capacity on the residual.** Slot 18 §0 shows Phoenix-Ops's −7.1141 (v7→v8) came from re-fitting the
  *identical* 123 columns with a 4,999-tree depth-9 CatBoost, adding **zero** features
  (`arrival_boost_contest.py:15, 61`; METHOD.md:157). **[S]** That belongs to **P7**, and P7's note that
  capacity "un-blocks feature families we rejected under a shallow model" applies directly here: if
  stages B or E fail, they should be re-run once, and only once, after P7 lands.
- **A LIRF-scoped 36 h variant fed to a gate-hold model.** OpenAir's −1.54 s from **383 rows**
  (status.md:1742) is the highest per-row leverage in either log **[S]**, but it requires the gate-hold
  decomposition itself, which is **P1** / **P5**.
- **CatBoost as a rival family** — **P10**.

---

## 9. Honest limits of this plan

**[X]** (1) I ran nothing. Every number is quoted from a file in this repository or from a slot report
that cites one. (2) The single most important number for this theme — gap-only at ten months — **does not
exist**, and no amount of reading will produce it; stage A is the plan. (3) My expectation that
`stand_arrival_taxi` underperforms OpenAir's −2.01 s rests on a gain-importance comparison
(`importance_all.json`) and on my reading of their `NUM0` list via slot 06; gain importance is a weak
proxy for ablation value and I have not ablated anything. (4) I did not re-derive the tie rate in the
arrival pool (R3) or the structural null floor that bounds the cap widening (§5); both are one-query
checks that stage 0 should run. (5) The EHAM/RASAS mechanism behind the reject-on-EHAM rule in stage D
is slot 18's hypothesis, consistent with our per-airport numbers but never tested against an EHAM
gap-length breakdown. (6) Effort figures are guesses.

**Sources**
- In-repo: `docs/competitor_research_2026-09-21/results/{18-turnaround-features,06-repo-openair,07-repo-phoenix-ops}.md`;
  `8bdf4f9:src/prc/features.py`, `8bdf4f9:tests/test_features.py`,
  `8bdf4f9:scripts/codex_fe/candidates.py`, `8bdf4f9:configs/codex_fe_gap.yaml`,
  `8bdf4f9:docs/codex_fe_results/{continuation_evaluation,continuation_state,continuation_protocol,protocol,screen,selection,joint_screen,parity,pipeline_parity,data_support,importance_all,refit_gap,refit_baseline_10m,refit_selected,refit_all,refit_noise}.{json,md}`;
  `HEAD:src/prc/{features,train,compare,cv,pipeline}.py`, `configs/main.yaml`, `Makefile`,
  `docs/README.md`, `docs/features_log.md`, `experiments/submissions.jsonl`.
- External, via slot 18 / 06 / 07: [Phoenix-Ops / zestful-fountain](https://github.com/Phoenix-Ops-LTD/prc2026-taxiout) `4477d9b`;
  [OpenAir / likable-eagle](https://github.com/ChandanHegde07/OpenAir) `97f603c`;
  [Schiphol RASAS 2025 (EN)](https://assets.ctfassets.net/biom0eqyyi6b/4WKbhQhXCHGBwnXhqSgWuQ/aca0896d2e7eb3311a15acbf0726ee07/Schiphol_RASAS_EN_2025_FINAL.pdf);
  [PRC Data Challenge 2026 — ranking dataset schema](https://prc-data-challenge-2026.netlify.app/data.html#the-ranking-dataset).
