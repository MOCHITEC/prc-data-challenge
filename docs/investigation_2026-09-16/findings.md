# Findings — investigation of 2026-09-16 (written 2026-09-17)

Synthesis of `plan.md` (§0 definitions, §2 verified negatives, §4 discrepancies) and all 30 files in
`results/`, read against `docs/handoff.md` and `docs/README.md` §2, §4, §6, §8, §10.

**Rules this file follows.** Every number is copied exactly as its result file states it. Every claim names the
result file it comes from. Measurements and the source agents' own inferences are kept apart. There are no
recommendations, no ranking by importance or usefulness, and no analysis that is not in a result file. Where two
result files (or a result file and a repository document) give different numbers for the same quantity, both are
shown and both files are named.

**Definitions** are `plan.md` §0 throughout: `y` = `TAXITIME_SEC_mvt`; `ms` = `mms` = `mvt_minus_sched`;
`proxy` = `MVT_TIME − AOBT_3`; "unmatched" = the routing definition (`mvt_minus_aobt3` null → model B;
22,470 training / 5,290 ranking rows) unless the FLIGHT_ID definition (22,424 / 5,283) is named; `f10` =
|BLOCK − SCHED| ≤ 10 s; 2 h = 7,200 s, 4 h = 14,400 s, 12 h = 43,200 s, 1 day = 86,400 s.

---

## 1. What was established

Grouped by what each mechanism explains. The order inside each group is not a ranking.

### 1.1 The label tail

| mechanism | deciding number | result file |
|---|---|---|
| The label is an exact timestamp difference | `TAXITIME_SEC_mvt == MVT − BLOCK` on **2,085,047 / 2,085,047** rows; `mvt_minus_sched == MVT − SCHED` on all rows; `(y − ms) == −(BLOCK − SCHED)` exactly on all rows | `sched-filled-block.md` §A |
| Therefore "sched-filled" and "y ≈ ms" are the *same predicate* | f10 ⇔ \|y − ms\| ≤ 10; D's `fill` and D's `y_eq_ms` differ only by threshold (10 s vs 60 s) | `sched-filled-block.md` §A, §4 |
| The fill is LIRF-specific | LIRF unmatched f10 **0.4845** vs ≤ 0.0565 at every other airport × group; given ms ≥ 2 h, LIRF **0.6316** (627 rows) vs non-LIRF **0.0002** (4,322 rows) | `sched-filled-block.md` §F, §G |
| …and it carries the 4–20 h band | label bucket 4–20 h: 106 rows, 98 % unmatched, **f10 0.943**; 2–4 h: 462 rows, f10 0.710; **> 20 h: 16 rows, f10 0.125** | `sched-filled-block.md` §C |
| The > 20 h band is a different mechanism: day rollover | **16** training rows y > 20 h; **14 of 16** have y − 86,400 ∈ [602, 1,992]; `date(MVT) − date(BLOCK) = 1 day` on **16 / 16**; 0 of 16 have a same-(ADEP, FLIGHT) departure on the BLOCK date | `block-day-rollover.md` §3.2, §3.3 |
| Rollover rows dominate the OOF metric | on `oof_main-48c19bc830`, the 10 rows with y > 20 h are **34.72 %** of total SSE (RMSE 51,121, pred median 37,131); removing them moves total RMSE **334.58 → 270.34** and unmatched **2,296.1 → 1,468.2** | `block-day-rollover.md` §3.5 |
| At LIRF & ms ≥ 12 h the label is two-valued | 29 rows: **18 filled (0.6207)**, **11 rollover (0.3793)**, 0 elsewhere; the 11 rollover labels are 87,170…88,392 (mean 87,536, sd 381) and are independent of ms (their ms is 50,581…93,535) | `cond-mean-vs-median.md` §3.1 |
| The ms-anchored mean is the best constant rule in every bin | ≥ 12 h bin, leave-one-month-out: `ms + mean(y − ms)` **12,715**, `ms` 14,100, constant bin mean 22,717, constant bin median 26,000, model **36,882**; 4–12 h: 5,855 vs 6,296 vs 11,706 vs 11,734 vs 7,951 | `cond-mean-vs-median.md` §3.4 |
| …and the median rule equals "predict ms" | median(y − ms) = **+4 s** (≥ 12 h), −1 s (4–12 h), −4 s (2–4 h); 3–4 s in every LOMO fold of the ≥ 12 h bin | `cond-mean-vs-median.md` §3.4 |
| LIRF unmatched rows are a distinct source | 1,487 / 1,488 `AIRCRAFT_TYPE_mvt` null (all other groups ≤ 0.31 %, matched groups 0.0); `FLIGHT_RULE_mvt` single value "I"; y median 3,922 vs 538–1,384 elsewhere; **439 / 584** of training y ≥ 2 h and **115 / 122** of y ≥ 4 h rows | `lirf-regime-tail-count.md` §3.1 |
| The correlation pre-check (E H2) separates LIRF from LFPG | corr(y, ms) on unmatched rows: **LIRF 0.8964** (n 1,488) vs **LFPG 0.0205** (n 3,773); restricted to ms ≥ 2 h, 0.9212 vs −0.0365; all ten airports 0.0205–0.8964 | `mms-mixture-reparam.md` §3.1 |
| Model B's large non-LIRF predictions are false alarms | 9 non-LIRF unmatched OOF rows predicted **14,436–16,880** with labels **549–1,431** (none ≥ 2 h, none sched-filled), together **2.60 %** of total SSE; non-LIRF unmatched labels for ms ∈ (14,400, 43,200] (368 rows) and (43,200, ∞) (46 rows) have sched-filled fraction **0.000** and y medians 1,019 / 1,099 | `false-alarm-non-lirf.md` §B, §C2 |
| Model B saturates | the baseline forward-fold model's largest unmatched prediction is **34,539.68** and **0** predictions exceed 43,200, while three scored rows have y ≈ 87.5 k and one has ms = 86,102 | `tree-identity-saturation.md` §4 (A) |
| `min_data_in_leaf` 50 exceeds the LIRF tail count | LIRF unmatched training rows with ms > 43,200: **29** on all 12 months, **25** on the forward training side; both < 50 | `tree-identity-saturation.md` §3.1 |
| The fill flag is not observable at prediction time | `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are null on **344,841 / 344,841** ranking rows; `mvt_minus_sched` null rate 0 | `sched-filled-block.md` §K; `block-day-rollover.md` §3.7 |
| LIRF matched rows have their own fill | LIRF matched f10 **0.1782** (28,369 rows; by month 0.145–0.206); on filled ∧ \|AOBT_3−SCHED\| > 600 s rows (6,453 rows = 4.1 % of LIRF matched) sd(resid) is **1,138.3** vs 252.6 on the near cell, and these 4.1 % carry **45.3 %** of Σresid²; in OOF 1,856 such rows have RMSE **1,105.6** = 3.01 % of total SSE | `lirf-matched-fill.md` §3.3, §3.7 |
| Matched proxy outliers exist and are two distinct patterns | **1,808** matched rows with \|BLOCK − AOBT_3\| > 1 h (10,554 > 30 min; 6.82 % > 10 min); LTFM 657 rows have AOBT_3 *earlier* than BLOCK (gap median +3,826, y median 850); EGLL 292/296 and LFPG 174/186 have BLOCK *earlier* (gap −4,616 / −5,344, y median 5,038 / 4,600). In OOF these 689 rows are **2.51 %** of total SSE | `matched-proxy-outliers.md` §3.1, §3.2 |
| Return-to-stand twins are real and exactly quantified | on the 98 same-take-off mixed pairs (all LFPG): `BLOCK_m < BLOCK_u` **98 / 98**, `y_m > y_u` **98 / 98**, and `y_m − y_u = (BLOCK_u − BLOCK_m) − (MVT_u − MVT_m)` with max \|difference\| **0.0**; y_m median 4,742.5 vs y_u 1,083 | `return-to-stand-twins.md` §3.3 |
| …but they are not the tail | twin rows are 28 of 584 training rows y ≥ 2 h, **1 of 122** y ≥ 4 h, and **0.366 %** of OOF SSE | `return-to-stand-twins.md` §3.4, §3.5 |
| The tail is not reachable by pooling matched rows | matched rows carrying the LIRF ∧ actype-null ∧ ms ≥ 2 h signature: **0**; pooling adds **103 / 3 / 1** rows at y ≥ 2 h / 4 h / 12 h; unmatched rows are 82.4 % / 97.5 % / 97.1 % of the pooled tail at those thresholds | `shared-b-no-transfer.md` §E, §H |

### 1.2 The leaderboard gap between builds

| mechanism | deciding number | result file |
|---|---|---|
| The v9 probe is an exact identity | v9 equals v8 on all **339,551** matched rows and v4 on all **5,290** unmatched rows | `lirf-regime-tail-count.md` §3.4 |
| …so the v4 → v8 LB gap is located exactly | SSE(v8) − SSE(v4) = (343.7964² − 305.027²)·344,841 = **8.674e9**, of which the matched part = SSE(v9) − SSE(v4) = **3.012e8 (3.5 %)** and the unmatched part = SSE(v8) − SSE(v9) = **8.373e9 (96.5 %)** | `lirf-regime-tail-count.md` §3.5 |
| The disagreement itself is LIRF and a handful of rows | RMS(v8 − v4) on unmatched = **1,735.5**; LIRF share of Σgap² **90.3 %**; top-20 share **80.4 %** (18 LIRF + 2 EGLL); RMS on the 1,721 ms ≥ 2 h rows 2,967 vs 469 elsewhere; matched RMS 73.8 | `lirf-regime-tail-count.md` §3.4 |
| v4 tracks ms on the ranking LIRF tail; v2/v7/v8 do not | on the 11 ranking LIRF rows with ms ≥ 12 h, v3/v4 predict 50,364–108,552 (≈ ms on 9 of 11) while v2/v7/v8 predict 28,965–40,267 on 9 of them and 66,037–68,581 on the two 16R rows; median v4 − ms = +3,907, median v8 − ms = −10,218 on the top-20 disagreement rows | `lirf-regime-tail-count.md` §3.4 |
| The v2 → v8 model-B difference is the two explicit refit seed keys | a 658-round refit with `seed` only reproduces `outputs/v2_two_clock.parquet` **on all 5,290 rows** after clip+rint; the same refit with `bagging_seed`/`feature_fraction_seed` reproduces `outputs/det_a.parquet` (v8) **on all 5,290 rows**; the two differ on **0 of 5,290** rows, RMS **327.8** raw / **314.5** in submission space; run 1 vs run 2 of the old form is bit-identical (max abs diff **0.0**) | `refit-seed-semantics-changed.md` §3.1, §3.2 |
| …and it is 2 % of the v4 gap, not its cause | RMS(a − v4) **1,748.5**, RMS(b − v4) **1,735.5**; the seed form is closer to v4 at 6 airports and further at 4; neither emits more than 2 predictions > 43,200 against v4's 18 | `refit-seed-semantics-changed.md` §3.2, §3.3, §4 (F) |
| The seed-key difference is spread, not concentrated | concentration of Σ(a − b)²: top-20 rows **18.7 %**, LIRF **43.5 %**, ms ≥ 2 h rows 62.7 % — against 80.4 % / 90.3 % for the v4-family gap | `refit-seed-semantics-changed.md` §3.2 |
| `hour_utc` does not produce the > 12 h predictions | the plan's own rule ("≥ ~15 confirms; 2 refutes") applied to the configuration the plan specifies returns **2**; all four crossed variants give 2; the largest ranking LIRF row (ms 111,654) is predicted **39,507** without `hour_utc` against v4's **108,552** | `hour-utc-absent-in-v3v4.md` §3.2, §3.4 |
| …and removing it moves *away* from v4 | RMS to v4: **1,802.3** without `hour_utc` vs **1,783.2** with it (v8-to-v4 distance 1,735.5); `hour_utc` carries **0.24 %** of the early-stop fit's gain | `hour-utc-absent-in-v3v4.md` §3.1, §3.3 |
| `submitted_v4.yaml` does not rebuild v4 | five resolved differences on today's code: model B leaves **127 → 255**, min_data **50 → 20**, rounds cap **1500 → 3000**, `force_row_wise` absent → True, `hour_utc` absent → present; plus the refit seed keys | `v4-config-not-reproducible.md` §3.2, §3.3 |
| …and the v4 build was never logged | no `experiments.jsonl`, `pipeline.jsonl` or `outputs/` record for v3 or v4 exists in any commit; `git log --all -S'v4_linear_tree-'` returns nothing; `docs/07_experiments.md` @957c913 lists v3/v4 CV as "—" | `v4-config-not-reproducible.md` §3.1; `cache-fingerprint-gaps.md` §5; `n-features-72-mislabel.md` §5 |
| The encoding leak does not reach the shipped model | removing December from the holdout rows' encodings alone leaves `best_iteration` at **542** (unchanged); RMS between the 585-round and 643-round refits on the 5,290 ranking rows is **63.3** overall, and the counts above 4 h (35) and 12 h (2) do not move | `holdout-encoding-self-inclusion.md` §3.2, §3.3 |
| Implied unmatched RMSE per submission (matched fixed at 218; 226 in brackets) | v1 2,174 (2,120), v2 1,816 (1,752), v3 1,751 (1,684), **v4 1,736 (1,669)**, v5 1,961 (1,902), v6 2,228 (2,176), v7 2,145 (2,091), **v8 2,157 (2,104)**, v9 1,753 (1,686) | `lirf-regime-tail-count.md` §3.5 |
| The library-version explanation was already dead | lightgbm 4.7.0, polars 1.44.2, numpy 2.4.6, pyarrow 25.0.1 dist-info all dated 2026-09-10 07:25, before v1–v4 | `plan.md` §2 `library-drift-refuted` (P 12) |

### 1.3 The evaluation machinery

| mechanism | deciding number | result file |
|---|---|---|
| `verdict()` reads two rows of the curve | it reads only the `dropped == 0` and `dropped == 200` entries (compare.py 100, 110); its signature has no month argument; `month_bootstrap_rmse_diff` is computed and printed (main, 157–161) and **never passed** to `verdict` (called at 169) | `decision-rule-blind-to-tail.md` §3.1 |
| The K = 200 drop set is the tail | det_a → ab_split at K = 200: **9 of the 10** rollover rows, **129 of 537** LIRF-unmatched rows, **51 of 233** LIRF sched-filled rows, **17 of 18** y ≥ 12 h rows, 189 unmatched vs 11 matched; the dropped 200 carry **53.9 % of SSE_A, 42.9 % of SSE_B and 84.5 % of Σ(p_a − p_b)²**; all 20 largest-disagreement rows are gone by K = 200 (13 already at K = 20) | `decision-rule-blind-to-tail.md` §3.2 |
| The K = 200 number is a property of the drop rule, not of the models | same pair, same 200-row budget: **−0.27** (max se, current) / **+1.83** (A's worst) / **−11.18** (B's worst) / **−8.43** (largest label) / **−0.48** (largest disagreement) / −34.59 (200 random rows) | `decision-rule-blind-to-tail.md` §3.2 |
| The bootstrap itself is correct | on y = [0,0,0], p_a = [1,2,3], p_b = [1,1,1]: hand diff −1.160247 = `diff_rmse`; exhaustive enumeration of the 27 resamples gives P(diff < 0) = 26/27 = 0.9630 against the function's 0.9632; identical predictions → CI exactly [0, 0]; NaN rows dropped and counted | `decision-rule-blind-to-tail.md` §3.4 |
| Past verdicts depend on the rule | of 28 available pairs the current rule ADOPTs **2**; full-CI-only or month-only would additionally ADOPT both 255/20 model-B pairs (−34.58 / −36.60), main-63086 → main-48c19 (−5.81), dest_distance (−2.89) and **a pure noise column** (ff1_base → ff1_noise, −0.25 [−0.56, −0.00]); K=200-only would ADOPT 1 → 3 seeds (−2.58) and f8 | `decision-rule-blind-to-tail.md` §3.6 |
| Screening resolution | K = 200 CI widths **0.39–0.60 s** on forward-only pairs (half-width 0.20–0.30) and 0.24–1.82 s on 3-fold pairs; K = 0 widths 0.29–5.89 / 0.17–57.1 s | `decision-rule-blind-to-tail.md` §3.6 |
| No fold has the production shape | train-month counts **11 / 11 / 6 / 10**; maximum forward horizon **2** months against production's 1 and 7; no fold crosses a year boundary; no forward fold predicts a January | `fold-shape-vs-ranking.md` §3.1, §3.2 |
| `overall` silently drops `jul_fwd` | fold order `[jan, jul, jul_fwd, forward]`; `jul` and `jul_fwd` predict the same 190,713 rows and `pl.coalesce` keeps the earlier one, so **overall.n = 672,428** while **Σ fold n = 863,141**; in all six four-fold records `overall.rmse_full` equals pooled{jan, jul, forward} to 3 dp and differs from pooled{jan, jul_fwd, forward} by 7.6–14.0 s | `fold-shape-vs-ranking.md` §3.3 |
| `overall`'s composition | Jan-2025 22.9 % (leaky fold), Jul-2025 28.4 % (leaky fold), Nov+Dec-2025 48.8 % | `fold-shape-vs-ranking.md` §3.3 |
| A forward-only OOF compared with a 3-fold OOF is silently restricted | det_a (3-fold) vs feat_base (forward): **328,009** rows enter = 0.488 of A, months 202511 + 202512 only, no warning; jan-only vs forward-only → **0** rows → "no full-data comparison" | `decision-rule-blind-to-tail.md` §3.5 |
| Model B's rounds are decided by the holdout month | widening `inner_holdout_months` 1 → 2 (same data, code and seeds) moves `rounds_unmatched` **45 → 413**, a factor **9.2**; the refit scaling factor accounts only for 1.109 → 1.290, so `best_iteration` itself moved from ≈ 41 to ≈ 320 | `holdout-month-decides-rounds.md` §4.3 |
| The forward fold's holdout is the least extreme month | October (model B): **1,913** rows, **4** with y ≥ 4 h, **0** ≥ 12 h, max y **25,558**; September + October: 4,378 rows, 16 ≥ 4 h, 4 ≥ 12 h, max y 87,543 | `holdout-month-decides-rounds.md` §4.1 |
| The feature cache is consistent today and no recorded run was stale | `feature_code` 590ceb8f8170ff6b = sha256(features.py) at HEAD; `n_features` in **every one of 41** recorded runs equals the count implied by `features.py` at its recorded `git_sha` (72 / 90 / 93 / 94 / 97 / 98 / 99) | `cache-fingerprint-gaps.md` §3.4, §4 (F, G) |
| `n_features` was never the matrix width | logged 72 for v1/v2/v5/v8 while model A saw **80 + 10** and model B **65 + 10**; logged 99 for v6 (matrix **107 + 10**) and for v7 (matrix **98 + 10**) | `n-features-72-mislabel.md` §3.1, §3.3 |
| The submission files are well-formed | all seven: 344,841 rows in `submitting.parquet` order, `MVT_ID_mvt` Float64 / `TAXITIME_SEC_mvt` Int32 (np.rint, ≤ 0.5 s); v9 probe exact (RMS(v8 − v9) matched = 0.0, RMS(up_v4 − v9) unmatched = 0.0) | `plan.md` §2 `alignment-weather-probe-checked-ok` (P 11) |

### 1.4 Distribution shift

| mechanism | deciding number | result file |
|---|---|---|
| The LIRF regime is at the same aggregate density in the ranking set | ranking Jan + Jul 2026: **383** LIRF unmatched rows vs **397** in training Jan + Jul 2025; ms ≥ 2 h **181 vs 181**; ms ≥ 4 h **43 vs 45**; ms ≥ 12 h **11 vs 11** | `lirf-regime-tail-count.md` §3.2 |
| …but not month by month | Jan 2026 **107** vs Jan 2025 **60** (0.97 % vs 0.53 % of LIRF departures; ms ≥ 2 h 52 vs 24); Jul 2026 **276** vs Jul 2025 **337** (1.74 % vs 2.22 %); training months range 52–337 | `lirf-regime-tail-count.md` §3.2 |
| The per-month OOF error tracks the LIRF-unmatched count | LIRF-unmatched SSE share by OOF month: Jan 17.9 %, Jul 49.6 %, Nov 45.8 %, Dec 31.1 %; per-month RMSE 373.0 / 378.4 / 291.5 / 277.9 becomes 338.0 / 268.6 / 214.5 / 230.6 without those 60 / 337 / 52 / 88 rows | `lirf-regime-tail-count.md` §3.3 |
| EHAM January 2026 contains a multi-day event | unmatched departures **805** (2026) vs **293** (2025); per day Jan 2–6, 9, 29 = 94 / 108 / 123 / 84 / 68 / 49 / 69 against a monthly median of 9; proxy sd on Jan 3–6 **1,502 / 1,963 / 1,848 / 1,644** vs Jan 2025's 220–665 on 28 days | `eham-jan2026-disruption.md` §3.1 |
| …with no training analogue in the *combination* | the highest training proxy sd on any airport-day is **1,118** (EGLL 12-31); no training airport-day combines proxy sd ≥ 1,400 with proxy median ≥ 1,500; by unmatched count alone there are 11 training days ≥ 68, including EHAM Sep 15–16 2025 (104, 71) | `eham-jan2026-disruption.md` §3.3 |
| …and it co-occurs with frozen weather | each flagged day has 6–17 frozen-precipitation hours (normal days 1.2 h mean); departures at 34–83 % of the same-weekday median; departure-row `deicing_gate` 0.417 on disruption-day unmatched rows vs 0.071 on normal days | `eham-jan2026-disruption.md` §3.1, §3.6 |
| Jan 2026 weather exposure exceeds the training year at three airports | ranking gate rows Jan 2026 **6,729** vs Jan 2025 **2,505** (×2.70 share); one ranking month has more gate rows than the whole 2025 training set at **EDDF (1,223 vs 617)** and **LSZH (1,379 vs 1,119)**; Jul 2025 and Jul 2026 have **0** gate hours at every station | `seasonal-deicing-shift.md` §3.4 |
| Gate hours are harder for the model | matched gate rows RMSE **345.7** vs 215.2 (×1.3–2.6 at every airport with gate rows); not a tail artefact (removing y > 7,200 changes the figures by ≤ 41 s); the pooled bias is small (+8.2 s) but is +121.8 at EGLL (249 rows) and +201.6 at LTFM (81 rows) | `seasonal-deicing-shift.md` §3.2 |
| The weather feature pays for itself on those rows | paired on the 328,009 forward-fold rows, no-weather → weather: matched gate rows **344.99 → 336.99** with mean residual **−61.0 → −8.7**; matched overall 203.35 → 202.48; unmatched 2,153.33 → 2,116.25 | `seasonal-deicing-shift.md` §3.7 |
| Schedule-anchored features shifted; clock-anchored ones did not | pooled matched ms median Jan **1,267 → 1,451**, Jul **1,684 → 1,503**; `aobt3_minus_sched` Jan 240 → 420, Jul 720 → 481; proxy medians move ≤ 47 s in 16 of 20 airport-months (max +64); congestion counts and headway medians are identical pooled and differ by ≤ 2 counts / ≤ 9 s per airport | `sched-anchored-delay-shift.md` §3.1, §3.2, §3.3 |
| The bin the rows fall into is where matched error lives | matched OOF RMSE by ms bin rises monotonically **119.6 (0–15 min) → 167.9 → 257.0 → 387.2 → 398.2 (> 120 min)**; Jan 2026 moves 7.9 pp of matched rows into ms ≥ 30 min against Jan 2025, Jul 2026 moves 7.0 pp out | `sched-anchored-delay-shift.md` §3.4, §3.5 |
| The unmatched population is not a different population | every rank-vs-train total-variation distance is **below** the Jan26-vs-Jul26 distance (airport 0.079 vs 0.206; prefix-2 0.197 vs 0.308; hour 0.048 vs 0.128; seen 0.125 vs 0.249) and comparable to the training year's own Jan+Jul-vs-all distance (0.081 / 0.136 / 0.070 / 0.054) | `unmatched-composition-shift.md` §3.2 |
| The rate gap is month composition | training 1.078 %, ranking 1.534 %, but the same-calendar-month training rate is **1.560 %**; Jan 2026 1.606 % vs Jan 2025 0.934 % (×1.72), Jul 2026 1.477 % vs Jul 2025 2.065 % (×0.72) | `fold-shape-vs-ranking.md` §3.4; `unmatched-composition-shift.md` §3.1 |
| The "seen" drop is not specific to unmatched rows | the same overlap statistic on **matched** ranking rows: **98.05 % (Jan 2026)** and **63.61 % (Jul 2026)**; ranking unmatched rows seen among matched *ranking* flights = **81.66 %**, the training within-set rate (81.90 %) | `unmatched-composition-shift.md` §3.3 |
| Composition reweighting is decided by fewer than ten rows | expected unmatched RMSE under the ranking (airport × month) mix **2,394.2 → 2,624.0 (×1.096)**; after dropping the top-10 OOF unmatched rows **×0.993**, top-20 ×0.970, top-50 ×1.000; 92 % of the expected MSE sits in three cells whose MSE is set by 1–13 rows | `unmatched-composition-shift.md` §3.5 |
| Timestamp precision is a 13-value grid, not rounding | at EDDM/EGLL/LFPG/LIRF/LTFM, MVT seconds occupy only **{54…59, 0…6}** — 13 distinct values, each of 55–59 and 0–5 carrying 0.0827–0.0836 (1/12), second 54 carrying 0.0629 and second 6 carrying 0.0208; at the other five airports all 60 values are present at 0.0162–0.0175 (1/60). **100 %** of cluster rows are within 6 s of a minute boundary | `timestamp-precision-clusters.md` §3.2 |
| …and the grid is unchanged in 2026 | ranking MVT s0: cluster 0.0813–0.0870, rest 0.0151–0.0184 | `timestamp-precision-clusters.md` §3.3 |
| The quantisation noise floor is small | 60/√12 = **17.32 s** per stamp, 60/√6 = **24.49 s** for a difference of two; adding in quadrature moves RMSE 217.7 by +0.69 s (+1.37 s triangular) and 334.58 by +0.45 s (+0.90 s) | `timestamp-precision-clusters.md` §3.8 |
| Adversarial separability, not re-measured | AUC **0.807** between 2025 training and 2026 ranking; reweighting did not change verdict ordering | `docs/handoff.md` §5, carried in `plan.md` §1 #17 (no result file re-measures it) |

---

## 2. What was refuted

### 2.1 Claims that come from the repository's own documents

| # | claim | where it currently lives | what killed it |
|---|---|---|---|
| R1 | **"極端例が 33 個 → ~2,000 個になる"** (shared_b would multiply the extreme examples) | `docs/README.md` line 237 (§10, Phase 2' candidate row a) | Pooling the 2,062,577 matched rows adds **1** row at y ≥ 12 h (33 → **34**), **3** at y ≥ 4 h (119 → **122**), **103** at y ≥ 2 h (481 → **584**). No label threshold maps 33 unmatched rows onto ≈ 2,000 pooled rows (y ≥ 1 h gives 1,087 → 4,132; y ≥ 45 min gives 1,222 → 11,736). — `shared-b-no-transfer.md` §H, verdict A2 |
| R2 | **"特徴量列数: 72（初日から変更なし）。うちエンコーディング 9 列"** | `docs/handoff.md` line 38 (§2), repeated line 221 (§11 review prompt); `docs/README.md` line 211 (§9 item 3) | 72 = **71 frame numerics + `month`**, and contains **0** encoding columns (the 9 `ENCODING_COLS` are joined inside `predict_fold` after the frame is built). `month` is dropped at every commit. The matrix LightGBM saw was 80+10 (v1/v2/v5/v8), 97+10 (v3/v4), 107+10 (v6), 98+10 (v7 and today) for model A, and 65 / 82 / 92 / 83 + 10 for model B — re-derived by executing each commit's own code. — `n-features-72-mislabel.md` §3.1, §3.2, verdict (g) |
| R3 | **"未マッチ行だけの対比較は 95% CI 幅が 62 秒"** and the conclusion **"未マッチ行だけを動かす変更は測れない … 触らない"** | `docs/handoff.md` line 82 (§5), line 241 (§11); `docs/README.md` line 106 and line 132 (§6 resolution table, "61.97 秒") | Refuted **on every available pair**: unmatched-only paired CI widths are **385.0** (det_a → ab_split), **724.4** (ab_single → ab_split), **421.5** (main-ab56551b5e → main-f9d57a661f), **5.57** (a forward-only pair) and **0.00** (det_a → det_b, whose model-B predictions are identical); matched widths 0.14–1.17. No available pair reproduces 62 s / 0.77 s, and the pair H measured is not identified in `docs/handoff.md`. — `decision-rule-blind-to-tail.md` §3.3, verdict (c) |
| R4 | **"model B の抽選性 … `rounds_unmatched` が 199 / 212 / 585 / 658 と乱高下し … その振れ幅が ±37 秒"**, and the remedy **"model B のシードを 10〜20 本平均"** | `docs/README.md` lines 188–190 (§8.1-2) and lines 237–238 (§10, Phase 2' row b) | Two separate refutations. (i) v2 and v8 have **identical** `rounds_unmatched` = 658, so round variation cannot be the v2 → v8 difference; the entire difference is the two explicit refit seed keys introduced in d182fc1, reproduced exactly (old form = v2 file on all 5,290 rows, new form = v8 file on all 5,290 rows). — `refit-seed-semantics-changed.md` §3.1, verdict (E). (ii) Seed averaging on model B was measured: unmatched OOF **2,312.5** at 3 seeds (main-ab56551b5e) vs **2,299.0** at 1 seed (det_a); LIRF 7,253 vs 7,206. — `plan.md` §2 `seed-avg-no-effect-b` (M). The ±37 s itself is confirmed to sit on the unmatched rows (96.5 % of the SSE gap, §1.2) but no measurement attributes it to the round lottery. |
| R5 | **"学習 22,470 行のうち、極端値（y ≥ 4h 相当）は 33 行"** | `docs/handoff.md` line 142 (§6), line 267 (§11); `docs/README.md` line 188 (§8.1-2); `configs/main.yaml` header ("the ~33 day-rollover rows") | **33** is the count at **y ≥ 12 h** (43,200 s), not y ≥ 4 h. y ≥ 4 h on the 22,470 unmatched rows is **119** (481 at y ≥ 2 h; 15 at y ≥ 80,000). Of the 34 training rows with y ≥ 43,200 only **16** are in the rollover band (y > 20 h), of which 15 are unmatched. — `plan.md` §4-1 (P 5); `block-day-rollover.md` §3.1; `shared-b-no-transfer.md` §B |
| R6 | **"early stopping 用ホールドアウトは約 1,650 行。そこに含まれる極端値は 0〜5 行"** | `docs/handoff.md` line 143 (§6), line 268 (§11); `docs/README.md` line 189 | The December holdout is **1,649** rows with **6** at y ≥ 14,400 (P 5; the 1,649 count is reproduced by `holdout-encoding-self-inclusion.md` §2, the extreme count is not re-measured — see §4 item 12). The October holdout the forward fold uses is **1,913** rows with **4** ≥ 4 h, **0** ≥ 12 h, max y 25,558. — `holdout-month-decides-rounds.md` §4.1 |
| R7 | **"v5 の予測は、12 時間超の taxi 時間を学習データの出現頻度の 2 倍の頻度で出していた"** | `docs/handoff.md` line 147 (§6), line 271 (§11); `configs/main.yaml` header lines 6–7 | Measured on the 5,290 unmatched rows, `v5cand` emits **2** predictions > 43,200 against a training rate of 33/22,470 ⇒ **~7.8 expected** — roughly a quarter of the training frequency, not twice. v2 2, up_v3 16, up_v4 18, v7 2, v8 2. Counting all 344,841 rows: 3 / 17 / 19 / 3 / 3 / 3 against 5.6 implied (all-month rate) or 14.0 (Jan+Jul rate). — `plan.md` §4-5 (P 3); `lirf-regime-tail-count.md` §3.4; `tail-postprocessing-cap-shrink.md` §3.9. **Caveat:** `outputs/v5_unmatched_capacity.parquet` is a *candidate* file; the uploaded v5 (`outputs/main.parquet` of 09-14 09:54) was overwritten, so the identity is unverified (`false-alarm-non-lirf.md` §5). |
| R8 | **"CV は本番を予測できるか → 条件付きでできる … 既知 2 事例（気象、model B 255 葉）で本番 LB と一致した"**, with the row `v2 → v6 (model B 255 葉): −34.82 / −0.06 [−0.61,+0.52] / −0.26` | `docs/README.md` lines 72–96 (§6) and line 165 (§8) | The v2 → v6 numbers are **not reproducible by any pair in `experiments/`** — the v2-era OOF file is not in the repository. The nearest available pairs give main-ab56551b5e → main-f9d57a661f (**−36.60 / −0.36 [−0.89, +0.18] / −0.25**) and det_a → ab_split (**−34.58 / −0.27 [−0.85, +0.31] / −0.24**). "The rule agreed with LB direction in two known cases" is recorded as **UNDETERMINED** (it needs the LB outcome of specific uploads). — `decision-rule-blind-to-tail.md` §3.6, §4, §5 |
| R9 | The rule described as **"採否は 200 行除外後の CI で決める"** | `docs/README.md` line 104 (§6) and line 159 (§8) | Refuted **as stated**, confirmed in a stricter form: the code REJECTs on the full CI *before* K = 200 is consulted (`det_a → main-48c19bc830` has K=200 −2.58 [−2.96, −2.15] and is REJECTed because the full CI is [−3.75, +0.55]). The rule as coded is "ADOPT only if the full CI **and** the drop-200 CI both exclude 0". — `decision-rule-blind-to-tail.md` §3.1, §4 |
| R10 | **"v5 は 255 葉 / min_data 20 で提出し、LB は 312.02 → 325.22"** read as "255/20 cost 13 s" | `docs/handoff.md` line 146 (§6); `configs/main.yaml` header lines 4–5 | The comparison crosses a code boundary: v5 (325.22) was built on new code while v2 (312.02) was built on old code. Against the same code family, v5 is 17–19 s **better** than v7 (342.59) / v8 (343.80). — `plan.md` §1 #29 (M `v5-verdict-confounded`, P 1). Settling it needs a same-code probe submission (Batch D, not run — §5). |

### 2.2 Hypotheses the investigation knocked down

| # | hypothesis | what killed it | result file |
|---|---|---|---|
| R11 | Item 22: v3/v4's model B lacked `hour_utc`, **and that absence** produced the > 12 h LIRF predictions | The plan's own decision rule ("≥ ~15 confirms; 2 refutes") returns **2** on the exact configuration specified; the largest ranking LIRF row is predicted 39,507 without `hour_utc` against v4's 108,552 | `hour-utc-absent-in-v3v4.md` §4 (B) |
| R12 | Item 21 extended: the seed-key change explains the v4 → current-code regression | Both seed forms sit **1,748.5 / 1,735.5** from v4 — the change moves ~2 % of a 1,736 s gap, and not systematically in either direction | `refit-seed-semantics-changed.md` §4 (F) |
| R13 | Item 13 / E H6: post-processing model B's tail (cap, or shrink toward the airport-hour median) improves RMSE | Every fixed cap ≤ 43,200 on unmatched predictions **worsens** RMSE (2,296.1 → 3,567.6 / 3,373.4 / 3,069.7 / 2,305.8); the cap-quantile curve is **monotone** (no q < 1.0 beats the baseline); every shrink λ ∈ {0.25, 0.5, 0.75} worsens it under **all seven** target definitions tried; the (airport, hour) target differs from the airport target by ≤ 35 s while the shrink itself costs 180–850 s. Capping *matched* rows also worsens (216.4 → 243.7 at 3,600) | `tail-postprocessing-cap-shrink.md` §3.2–§3.7 |
| R14 | Item 10 (C): `unmatched_min_data_in_leaf` 5 restores pred/ms ≈ 1 on LIRF | The sched-filled median pred/ms moves **0.7893 → 0.7473**, i.e. *away* from 1; only the ceiling moves (max 34,540 → 68,801; 4 predictions above 12 h where the baseline had 0) | `tree-identity-saturation.md` §4 (C) |
| R15 | Item 23 (E) / `docs/handoff.md` §9 「model B の early stopping を廃止して木の本数を固定する」: model B's rounds can be fixed from a config | `unmatched_num_rounds` is a **cap inside `min(...)`**, never a fixed count; `inner_holdout_months` / `early_stopping` are shared with model A; `unmatched_num_rounds: 600` produced a **bit-identical** run (`make compare` `+0.00 [+0.00, +0.00]` at all four drop levels) | `holdout-month-decides-rounds.md` §2, §4.2, §5 (E) |
| R16 | Item 11 (A): `use_linear_tree: true` can be applied to model B alone | `use_linear_tree` is one shared `TrainConfig` field; measured consequence of the run: `rounds_matched` **939 → 231** and matched RMSE **201.6633 → 202.5132**, so model A changed too | `linear-tree-model-b.md` §2, §5 |
| R17 | Item 24 (B): the encoding self-inclusion changes `best_iteration` | Removing December from the holdout rows' encodings alone leaves `best_iteration` at **542** (unchanged) and makes the December holdout **1.97 s worse**, not better (1,694.16 vs 1,692.19 with the booster fixed) | `holdout-encoding-self-inclusion.md` §4 (B, C, D) |
| R18 | Item 23 (D′): the round-count effect is confined to the tail / to LIRF | At 413 rounds, non-LIRF unmatched ms ≥ 2 h RMSE gets **worse** (2,132.29 → 2,235.14, max prediction 16,880 → 24,035), RMSE clean worsens (224.58 → 228.03), and the whole fold is **+2.75 s [+1.14, +4.32]** at K = 20 | `holdout-month-decides-rounds.md` §5 (D′) |
| R19 | Item 11: linear_tree leaves the matched side unharmed | matched RMSE **+0.85 s** (201.6633 → 202.5132) at 231 instead of 939 rounds; the whole-fold −53.58 s mixes a model-A regression with a model-B gain | `linear-tree-model-b.md` §5 |
| R20 | Item 9 (b′) / E H2: the `T = D − G` reparameterisation applied to **all** unmatched rows | pooled unmatched OOF **2,326.5 → 2,696.7 (+370.2)**; worse on jul and forward, worse on non-LIRF (1,451.7 → 2,179.4), EHAM 954.7 → 3,408.9 | `mms-mixture-reparam.md` §3.3, §4 |
| R21 | Item 9: a perfect fill flag would be enough | The **oracle** f10 → ms substitution gives only 2,203.3 (base 2,326.5) and 2,156.6 with a non-filled second component — both **worse than the learned mixtures**, because the 9 rollover rows are not f10 and keep their base predictions (RMSE 54,883, 59.8 % of base SSE). On the 129 f10 rows the oracle's RMSE is **4 s** | `mms-mixture-reparam.md` §3.3, §3.7, §4 |
| R22 | Item 12 / E H5: the public teams' rule overlays transfer | Rule Z (Phoenix-Ops clock overlay): OOF total **334.5821 → 334.6135 (+0.031)**, 19 rows improved / 25 worsened. OpenAir analogues (`pa` / `pl` / `ms` ≥ 7,200 → 50 % toward median): **+4.33 / +6.02 / +56.26**. Three of the four readings of the sergiuv11 LIRF overlay worsen it (+80.11 / +60.98 / +52.85). Rules Z, O-a and O-l can **never fire on a model-B row**: `AOBT_3` and `LOBT` are null on exactly the 22,470 / 5,290 unmatched rows | `rule-overlays-extreme-rows.md` §A, §D, §4 |
| R23 | Item 19 (S3) / E H10: OOF residual SD differs by (airport, MVT second == 0) | **Refuted for matched rows**: within-airport sd ratios **0.923–1.014** at all 10 airports; per month 40 pairs of ratios 0.78–1.12 with no consistent sign. The pooled gap (235.6 vs 215.3) is composition — s0 rows are 84 % cluster rows. The two large unmatched ratios (LFPG 4.29, LIRF 1.87) are **1 and 2 day-rollover rows** (the LFPG s0 cell falls 7,701.7 → **699.7** without one row) | `timestamp-precision-clusters.md` §3.4, §4 (S3) |
| R24 | Item 19 (S6): a second-of-minute indicator is a model feature | No `.dt.second` / `% 60` anywhere in `src/prc/`; no such column among the 98 numeric features. (It is algebraically present: SCHED seconds are 0 on every row, so `mvt_minus_sched mod 60 == 0 ⇔ MVT second == 0` on all 2,085,047 rows) | `timestamp-precision-clusters.md` §3.7 |
| R25 | Item 5 (sub-claim 9): the 22 ranking matched rows predicted ≥ 2 h are proxy outliers of the training kind | **20 of 22** have proxy 594–5,437 s, i.e. normal; only 2 have proxy ≥ 2 h. The training set is defined by a BLOCK − AOBT_3 gap, and BLOCK is 100 % null in the ranking set, so membership is untestable | `matched-proxy-outliers.md` §4 (9) |
| R26 | Item 17: ranking unmatched rows are "a different population" | Every rank-vs-train TV distance is below the Jan26-vs-Jul26 distance and within 1–2× the training year's own Jan+Jul-vs-all distance; the two large deviations (EHAM–KL in Jan 2026; "seen" in Jul 2026) are each one cell / one mechanism | `unmatched-composition-shift.md` §4 |
| R27 | Item 7 (as worded): "the ranking set contains many such rows" (rows on which model B fires 15–20 k) | Exactly **1** non-LIRF unmatched ranking row is predicted ≥ 4 h by any submission (EGLL, ms 25,438: v2 15,608, v8 12,299, v7 12,193, **v4 100**); **no** submission predicts ≥ 12 h on any non-LIRF unmatched ranking row; maximum non-LIRF prediction 15,608 (v2), 12,700 (v8), 5,694 (v4) | `false-alarm-non-lirf.md` §E, §4 |
| R28 | Item 7 (mechanism, as worded): model B fires 15–20 k on large-ms non-LIRF rows generally | Confirmed for the ms ≥ 12 h bin only; in the 4–12 h bin **0 of 143** OOF rows are predicted ≥ 4 h (median 1,022, and the **p10 is 0** — an under-prediction failure in the same bin) and in 2–4 h **0 of 1,673** | `false-alarm-non-lirf.md` §C, §4 |
| R29 | Item 1 (as worded): "the extreme tail is this artefact [sched-fill]" | Confirmed for the 4–20 h band (f10 0.943 on 106 rows); **refuted for > 20 h** (f10 0.125 on 16 rows, 75 % with BLOCK more than 1 h before SCHED) | `sched-filled-block.md` §4 |
| R30 | M's "None at non-LIRF except two LFPG rollovers" (unmatched y ≥ 4 h) | There are **4** non-LIRF rows: LSZH 87,341, LFPG 84,240, LFPG 58,206, LSZH 18,409 — and the two rollovers are **one LSZH and one LFPG**, not two LFPG. None of the 4 is sched-filled | `sched-filled-block.md` §E, §4 |
| R31 | M's "8 rollover rows ≈ 25 % of total SSE" | Those 8 rows are **30.9 %** of total SSE; **25.5 %** is the share of the 9 rows with y ≥ 86,400 | `block-day-rollover.md` §4 |
| R32 | D's "y − 86,400 = 602…1,992 for 13 of 16" | **14 of 16** fall in [602, 1,992]; the other two are −2,160 (the LFPG row) and 44,767 (a row that is also sched-filled) | `block-day-rollover.md` §4 |
| R33 | Item 6: twins explain the extreme tail; "take-off within 2 min" describes the 450 training rows | Twin rows are 28 of 584 y ≥ 2 h rows, **1 of 122** y ≥ 4 h, 0.366 % of OOF SSE. Only **103 of 223** pairs have take-offs within 120 s, and **all 103 are at LFPG**; the other 106 pairs are hours apart (median 2.9–8.4 h). Of the 84 ranking rows, only **32 (16 LFPG pairs)** fit the pattern. D's "4 FLIGHT_ID overlaps" are ranking **ARR** rows of year-boundary flights; DEP-vs-DEP overlap is 0 | `return-to-stand-twins.md` §3.2, §3.4, §3.6, §3.8 |
| R34 | Item 4 (causal wording): the fill is what makes LIRF the worst matched airport | Removing all filled rows lowers LIRF matched OOF RMSE 347.1 → **294.4**, but LIRF **stays first** (next is LFPG 247.8). It is the largest single contributor, not a sufficient cause: the not-filled ∧ far cell (21,380 OOF rows, RMSE 355.7) carries **45.5 %** of LIRF-matched SSE against the filled ∧ far cell's 38.1 % | `lirf-matched-fill.md` §4 (S6) |
| R35 | Item 28 (F): some recorded experiment ran on a stale feature cache | Refuted for **all 41** recorded runs: `n_features` in every record equals the count implied by `features.py` at its recorded `git_sha`, including the five runs whose `git_sha` was committed 31–61 s after the estimated start | `cache-fingerprint-gaps.md` §4 (F) |
| R36 | P 7: "`n_features` semantics changed at 97e67c6" | Refuted as stated: 97e67c6 introduced `visible_numeric_columns` but `run_cv` still logged the **frame** width there (`main-48c19bc830` @ 97e67c6 logs 99 while the model saw 98). The log expression changed one commit later, at **fd848bf**, and **no record with the new semantics exists** (`grep -c n_frame_columns experiments/experiments.jsonl` = 0) | `n-features-72-mislabel.md` §4 (f) |
| R37 | P: "top-20 rows carry 73.2 % of Σgap²; all 20 are LIRF" (v8 − v4) | **80.4 %**, and **18 LIRF + 2 EGLL** (the two EGLL rows have ms 25,438 / 13,199 with v4 100 / 1,603 against v8 12,299 / 12,700) | `lirf-regime-tail-count.md` §4 (see §4 item 3 below) |
| R38 | Item 3 (D) as a general statement: "differences between submissions are decided by those rows" | Confirmed only for the v4-family-vs-v2/v7/v8-family difference (LIRF share 89–90.5 %, top-20 share 80–82 %). For v3 − v4, v8 − v2, v7 − v8 and v8 − v5cand the LIRF shares are **24–47 %** and top-20 shares **19–36 %**; the 32 s v2 → v8 LB change comes with **no** LIRF-tail prediction change | `lirf-regime-tail-count.md` §4 (D) |
| R39 | Item 26 / P 8: `cv.py` "96–121" for the fold builder; the plan's `compare.py` line citations | At HEAD the folds are at **cv.py 26–51**; `compare.py` has **175 lines** and `cv.py` **125**, so "518–546 / 470 / 584" and cv.py "157–187" do not exist (the functions are at compare.py 34–61 / 64–88 / 91–119 / 122–129 / 132–171 and cv.py 87–117). `train.py` 329/348 is confirmed | `fold-shape-vs-ranking.md` §4; `decision-rule-blind-to-tail.md` §3.1 |
| R40 | Item 16 / E H11: "2–4× more de-icing-gate hours" as a general statement; "Jan/Jul labels elevated vs mid-year" | The ×2–4 holds at EHAM (×2.50), LSZH (×3.05), EDDF (×3.65) and LTFM (0 → 71 h), but **not** at EDDM (×1.63), LFPG (×0.48), EGLL (×0.12), LEBL/LIRF (0/0); Jul 2026 is **0 everywhere**. Jan 2026 exceeds every 2025 month at EDDF/EHAM/LSZH/LEMD (marginally EDDM) but **not at LTFM** (Feb 2025 0.1756 > Jan 2026 0.0954). For labels: Jul is rank 1 on mean and SD, but Jan's mean (995) is rank **4.5 of 12** (= Dec 995, < Nov 997, < Feb 1,002); Jan's SD (605) is rank 3 (Feb 628 is higher) | `seasonal-deicing-shift.md` §4 |
| R41 | Item 18: "proxy medians stable (≤ 50 s)" as a threshold; "taxi-in means show no shift" | 4 of 20 airport-months exceed 50 s (max +64 s, EHAM Jul). `arr_taxi_in_mean_30m` median moves pooled Jul **−37.4 s**, EHAM Jul **−150.3 s**, EDDF Jul −69.0 s | `sched-anchored-delay-shift.md` §4 |
| R42 | Item 15 (D's day list): "Jan 2–7, 9, 29 with 68–123 unmatched/day" | Under the 3× rule the days are **Jan 2–6, 9, 29** (7 days, 595 unmatched rows). **Jan 7** (18 unmatched) does not pass the count rule, although it matches the flagged days on proxy sd (1,440), proxy median (1,573) and departure shortfall (34 % of the weekday median). **Jan 9** has 49, below the stated 68–123. D's "Jan 2025 proxy sd 220–600" omits Jan 5 (1,036), Jan 18 (869) and Jan 11 (665) | `eham-jan2026-disruption.md` §4 (B), (A) |
| R43 | Item 31 (B5) unqualified: "the LIRF actype-null sched-filled regime does not exist among matched rows" | Confirmed for the actype-null signature (**0** matched rows), but the *fill mechanism* BLOCK = SCHED with ms ≥ 2 h does occur in **32** matched LIRF rows (all y ≥ 2 h, y = ms, 28 with \|BLOCK − AOBT_3\| > 1 h) — 1.2 % of the 2,751 matched LIRF ms ≥ 2 h rows against 63.2 % of 627 unmatched | `shared-b-no-transfer.md` §4 (B5) |
| R44 | Commit 5b90919's premise: the matched rows "carry the same airport/hour/aircraft structure" | Confirmed for airport / hour / runway / stand / traffic-window / weather columns (66 `cols_b` numerics + 5 categoricals ≥ 99.8 % non-null on both groups), **refuted for aircraft and operator**: `AIRCRAFT_OPERATOR_flt`, `MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` are null on **100 %** of unmatched rows (training and ranking), and `AIRCRAFT_TYPE_mvt` is null on 1,487 / 1,488 LIRF unmatched rows and **0** matched rows | `shared-b-no-transfer.md` §4 (A1) |
| R45 | The code comment at train.py:299 ("One seed is the default, and then this is the single fit it was before") | **False as written**: with `n_seeds` = 1, 0 of 5,290 predictions are equal between the two forms | `refit-seed-semantics-changed.md` §4 (B) |
| R46 | Item 14: "the tail is predictable signal" as a blanket statement | Confirmed for the sched-filled component (L2 reaches RMSE 5,694 with pred/ms 0.821 on the 129 f10 rows; variant A gets it to 2,158) but **refuted for the rollover component**: even L2 has RMSE **54,883** on the 9 rollover rows and they still carry 59.8 % of its SSE — L2's advantage there is that it over-predicts them less badly, not that it predicts them | `robust-loss-harmful.md` §4 (d′) |
| R47 | Item 14: "the early-stopping metric explains the huber result" | huber α = 5,000 with `metric: l2` gives 2,458.1 vs 2,481.9 with huber's own metric — **24 s of 155**, on different round counts | `robust-loss-harmful.md` §4 |
| R48 | Item 30 / E H7 / `docs/handoff.md` §8 (A4): training model B on all ~2.08 M rows **improves unmatched predictions** | Unmatched RMSE **2,109.32 → 2,144.73** (89 columns) / **2,154.29** (93 columns, the literal removed `shared_b`); whole fold **284.7038 → 287.1174 / 287.7726**. The full-data difference is **+35.41 [−409.48, +275.50]** / **+44.98 [−397.22, +284.72]**, i.e. a non-significant *worsening*, and `prc.compare.verdict` returns **REJECT** for both. (Its stated mechanism does hold off the tail — see §4 item 2.) | `shared-b-all-rows.md` §3.1, §3.3, §4 (A4) |

### 2.3 Numeric claims in the docs the investigation did not decide, flagged for correction

- `docs/README.md` line 54 (§4) gives the leader's implied unmatched RMSE as **≈ 1,033**, while line 198 (§8.2)
  of the same file gives **≈ 929** for the same leader score (245.02). `lirf-regime-tail-count.md` §3.5
  recomputed the implied table (matched fixed at 218 and 226) for v1–v9 but has no leader row; the 929 figure is
  M's, carried in `plan.md` §1 #3. Nothing in `results/` decides between 1,033 and 929.
- `docs/README.md` line 24 and `docs/handoff.md` §1 give the standing as 53 / 127 teams, leader youthful-giraffe
  245.02, 2nd enthusiastic-daisy 249.20 (H, 2026-09-15). E's 2026-09-16 pull of `data.json` shows **154 teams**,
  vigorous-whistle **243.58** (13 subs), enthusiastic-daisy **244.83** (49), youthful-giraffe 245.02 (21, last
  09-10). See `plan.md` §4-10.
- `docs/README.md` §4 and `docs/handoff.md` §3 (y ≥ 2 h 210 rows 36.7 %; y ≥ 4 h 55 rows 33.6 %; y ≥ 1 day
  10 rows 24.1 %; unmatched 1,864.9 → 1,044 → 177.9) are **confirmed but on a different OOF file**: they
  reproduce exactly on **`oof_main-f9d57a661f`** (total 300.17, unmatched 1,864.9), not on
  `oof_main-48c19bc830` (total 334.58, unmatched 2,296.1) where the same 10 rows are **34.7 %**.
  — `block-day-rollover.md` §3.6, §4. The docs do not name the file.
- `docs/handoff.md` §2's 「AOBT_3 が分単位に丸められている割合 96.6%」 is confirmed **with its denominator
  identified**: 2,015,132 rows exactly at :00 = **0.9665** of all 2,085,047 training rows (nulls included),
  = **0.977** of the 2,062,577 non-null rows. — `timestamp-precision-clusters.md` §3.1

---

## 3. What moved model B

Three protocols are in play and their numbers must not be mixed.

**Protocol A — three-fold model-B refits (items 9 and 14).** `jan` + `jul` + `forward` validation rows =
1,435 + 3,938 + 3,005 = **8,378**, the exact row set on which the plan's 2,296.1 is defined. Baseline
reproduced **in this protocol's own run** at **2,326.5** against the OOF file's **2,296.1** (+30.4, +1.3 %;
per fold jan 3,247.5 / 3,223.0, jul 2,013.6 / 2,008.8, forward **2,177.4 / 2,109.3**). RMS(repro − file)
213.8 s, correlation 0.9967, 8,145 of 8,378 rows differ by more than 1 s. Paired: the file is better by
**30.5 s [−56.6, −2.2]** at K = 0 but **+2.3 [−13.3, +19.6]** at K = 20 and **+1.8 [−5.7, +8.5]** at K = 200 —
the whole difference is a handful of tail rows. Cause: `num_threads` 4 in the logged run vs 2 here; the visible
consequence is the round count (jan 393 → 566, jul 479 → 288, forward 45 → 45).
(`mms-mixture-reparam.md` §3.2; carried into `robust-loss-harmful.md` §2.)

**Protocol B — forward-fold pipeline runs (items 10, 11, 23).** 328,009 scored rows, of which **3,005**
unmatched. Baseline `inv_baseline_forward-67e76a5dfc`: total **284.7038**, matched **201.6633**, unmatched
**2,109.32**, `rounds_unmatched` **45**, `rounds_matched` 939. This baseline reproduces the OOF file's forward
figure (2,109.3), while Protocol A's reproduction of the same fold gives 2,177.4.

**Protocol C — post-processing the logged OOF file (items 12, 13, and item 14's clipping half).** No fits.
Baseline is the file itself: total **334.58**, unmatched **2,296.1** on 8,378 rows, matched 216.4, LIRF
unmatched 7,179.3, non-LIRF unmatched 1,450.2.

### 3.1 Protocol A — fitted variants (baseline 2,326.5)

Tail-trimmed columns use `prc.compare.robustness_curve` (500 draws, the K rows where *either* model errs most),
**applied to the 8,378 unmatched rows only** — so these items produce CIs, not a `make compare` VERDICT string.

| variant | unmatched OOF RMSE | delta (K = 0) and CI95 | K = 20 | K = 200 | survives dropping 20 / 200? |
|---|---|---|---|---|---|
| `oof_main-48c19bc830` (the file) | 2,296.1 | −30.5 [−56.6, −2.2] | +2.3 [−13.3, +19.6] | +1.8 [−5.7, +8.5] | no / no |
| **A — target y − ms on LIRF rows** | **1,632.1** | **−694.4 [−1,051.0, −381.8]** | **−185.1 [−261.6, −111.2]** | **−15.7 [−32.2, +2.3]** (P = 0.952) | **yes / CI includes 0** |
| A2 — target y − ms on all unmatched rows | 2,696.7 | +370.2 [−149.5, +1,012.1] | +51.7 [−66.6, +188.2] | +45.2 [+25.9, +63.6] | worse / worse |
| **B — P·ms + (1−P)·base** | **1,926.1** | −400.4 [−656.5, −168.4] | −22.5 [−68.8, +22.5] | **−25.0 [−41.5, −9.9]** | CI includes 0 / **yes** |
| B-hard — ms where P ≥ 0.5 | 1,961.7 | −364.8 [−652.1, −106.8] | +2.8 [−47.2, +51.4] | −49.4 [−63.6, −36.8] | no / yes |
| B-LIRF-only | 1,922.7 | −403.8 [−659.5, −171.9] | −12.7 [−54.4, +26.4] | −25.1 [−41.3, −10.0] | CI includes 0 / yes |
| **B2 — P·ms + (1−P)·non-filled regression** | **1,820.6** | −505.9 [−787.8, −243.4] | **−116.3 [−186.5, −51.0]** | −11.7 [−27.8, +5.2] | yes / CI includes 0 |
| B2-LIRF-only | 1,867.2 | −459.3 [−725.3, −208.3] | −32.6 [−77.6, +7.5] | −29.7 [−43.1, −13.7] | CI includes 0 / yes |
| oracle f10·ms + (1−·)·base | 2,203.3 | −123.3 [−189.3, −72.3] | −141.9 [−185.2, −103.4] | −67.9 [−80.3, −55.1] | yes / yes |
| oracle f10·ms + (1−·)·non-filled | 2,156.6 | −169.9 [−548.0, +149.2] | −361.4 [−438.2, −281.8] | −110.4 [−132.8, −88.7] | yes / yes |
| non-filled regression alone | 2,646.2 | +319.7 [+35.9, +618.3] | +280.8 [+162.6, +403.1] | +43.8 [+8.3, +84.1] | worse / worse |
| huber α = 0.9 (LightGBM default) | 3,764.6 | +1,438.0 [+1,003.6, +1,819.1] | +560.7 [+336.6, +784.2] | +104.4 [+55.8, +143.3] | worse / worse |
| huber α = 5,000 | 2,481.9 | +155.4 [−5.2, +297.4] | −167.9 [−257.8, −95.2] | −29.9 [−43.6, −16.8] | sign reverses at K = 20 |
| huber α = 5,000, `metric: l2` | 2,458.1 | +131.6 [−25.4, +273.7] | −151.4 [−233.1, −76.1] | −14.3 [−27.1, −1.6] | sign reverses at K = 20 |
| quantile α = 0.5 | 2,720.2 | +393.7 [+106.6, +652.7] | −202.8 [−306.7, −95.2] | −50.2 [−69.0, −31.1] | sign reverses at K = 20 |
| log1p target | 2,679.7 | +353.2 [+47.8, +620.9] | −192.7 [−305.8, −89.8] | −69.0 [−91.1, −44.4] | sign reverses at K = 20 |

Per-fold and per-airport detail (`mms-mixture-reparam.md` §3.3, §3.4; `robust-loss-harmful.md` §3.1, §3.2):
variant A improves **each of the three folds** (3,247.5 → 2,904.6; 2,013.6 → 1,338.6; 2,177.4 → 1,024.6), LIRF
(7,326.3 → 4,214.3) **and** non-LIRF (1,451.7 → 1,276.7), and **nine of ten airports** with none worse (EDDF
1,064.5 → 409.2, EHAM 954.7 → 477.2, LEBL 1,043.4 → 371.4). It removes **all 10** non-LIRF predictions ≥ 4 h
(10 → 0) while raising predictions ≥ 12 h from 4 to 15 against 17 such labels. The classifier behind B is
learnable: validation AUC **0.9567 / 0.9681 / 0.9701** per fold, monotone calibration across six P bins
(observed f10 rate 0.010 → 0.900), and P ≥ 0.5 occurs at LIRF 245 rows, LSZH 11, EHAM 2, every other airport 0.
Every robust loss **improves** the 8,215 rows with y < 2 h (1,097 → 689–888) and loses on the 163 with y ≥ 2 h
(14,750 → 16,458–26,347); huber α = 0.9 emits **no** prediction above 1,462 s and never early-stopped (best =
the 2,000-round cap in all three folds).

### 3.2 Protocol B — forward-fold runs (baseline unmatched 2,109.32, total 284.7038)

These items ran the repository's own decision rule on all 328,009 rows and produced a verdict. Item 30 is a
special case: model A is **not refitted** under `shared_b`, so its whole-fold figures combine each variant's
unmatched predictions with the logged matched OOF column (325,004 rows, RMSE 201.6633), and the report gives
both a whole-fold and an unmatched-only curve. Its baseline `S-full` reproduces the logged record **bit-exactly**
(unmatched **2,109.3167**, matched 201.6633, total **284.703845**, rounds 45, row-by-row max \|diff\| **0.0000**).

| variant | unmatched OOF RMSE | delta | `prc.compare` (full CI, K = 200) → VERDICT | survives dropping 20 / 200? |
|---|---|---|---|---|
| **item 10 — `unmatched_min_data_in_leaf: 5`** | **1,276.05** | **−833.27** (total 284.70 → 234.97, −49.73) | full −49.73 [−91.52, −13.99]; K=200 **+0.20 [−0.02, +0.42]** → **REJECT — tail-only** | no (K=20 **+1.13 [+0.17, +2.17]**) / no (K=2000 +0.05 [−0.02, +0.12]) |
| **item 11 — `use_linear_tree: true` (both models)** | **1,181.29** | **−928.03** (total 284.70 → 231.13, −53.58) | full −53.58 [−96.86, −17.04]; K=200 **+1.22 [+0.75, +1.62]** → **REJECT — tail-only** | no (K=20 **+1.37 [+0.22, +2.60]**) / no (K=2000 **+1.11 [+0.88, +1.33]** — the only Batch C run whose drop-2000 CI excludes zero on the worse side) |
| item 23 (a) — `unmatched_num_rounds: 600` | 2,109.32 | **0.00** (bit-identical run) | `+0.00 [+0.00, +0.00]` at all four drop levels → **REJECT — no difference** | n/a |
| item 23 (b) — `inner_holdout_months: 2` | 1,838.94 | −270.38 (total 284.70 → 266.92, −17.79) | full **−17.79 [−37.07, +2.03]** (spans 0); K=200 **+0.66 [+0.31, +0.99]** → **REJECT — no difference on all rows** | no (K=20 **+2.75 [+1.14, +4.32]**) / no (K=2000 +0.27 [+0.17, +0.35]) |
| **item 30 — `P-restr`: model B pooled on all 1,757,038 labelled rows, 89 columns** (the item's mandated variant) | **2,144.73** | **+35.41** (total 284.7038 → **287.1174**) | whole fold: full **+2.41 [−20.40, +20.49]**; K=200 **−0.83 [−1.17, −0.50]**; K=2000 **−0.15 [−0.24, −0.06]**; month +2.41 [−5.35, +9.77] → **REJECT — no difference on all rows** | unmatched-only curve: better at every K ≥ 20, and **at K = 200 and K = 2000 the CIs exclude zero** — K=20 −28.09 [−69.73, +11.58] (spans 0), **K=200 −25.37 [−36.36, −13.11]**, **K=2000 −11.61 [−15.19, −8.40]** (full CI **+35.41 [−409.48, +275.50]**, P(B) = 0.426) |
| **item 30 — `P-full`: the literal removed `shared_b`, 93 columns** | **2,154.29** | **+44.98** (total 284.7038 → **287.7726**) | whole fold: full **+3.07 [−19.68, +21.36]**; K=200 **−0.93 [−1.24, −0.64]**; K=2000 **−0.17 [−0.26, −0.08]**; month +3.07 [−4.36, +10.13] → **REJECT — no difference on all rows** | unmatched-only curve: same shape — K=20 −13.90 [−59.31, +34.26] (spans 0), **K=200 −27.83 [−38.24, −18.05]**, **K=2000 −12.01 [−15.52, −8.32]** (full CI **+44.98 [−397.22, +284.72]**, P(B) = 0.404) |
| item 30 — `S-restr`: split fit, 89 columns | 2,109.3167 | **0.00** (bit-identical to `S-full`, max \|diff\| 0.0) | `+0.00 [+0.00, +0.00]` at all four drop levels → **REJECT — no difference** | n/a — LightGBM discards the four single-bin categoricals at Dataset construction in an unmatched-only fit |

Side effects measured in the same runs: item 10 leaves model A **identical to four decimals** (201.6633 both
sides, `rounds_matched` 939 → 939) and improves non-LIRF unmatched ms ≥ 2 h from 2,132.29 to **943.72**; item 11
changes model A (`rounds_matched` 939 → 231, matched **201.6633 → 202.5132**) and also improves non-LIRF
unmatched ms ≥ 2 h (2,132.29 → 1,208.95, max prediction falling 16,879.98 → 10,550.34); item 23 (b) makes
non-LIRF unmatched ms ≥ 2 h **worse** (2,132.29 → 2,235.14). Tail counts above 12 h / 4 h: baseline 0 / 17,
item 10 4 / 9, item 11 4 / 9, item 23 (b) 3 / 16. Maximum unmatched prediction 34,539.68 → 68,801.03 (item 10),
→ 76,605.44 (item 11), → 50,728.42 (item 23 b). On the 45 LIRF sched-filled rows with ms ≥ 2 h, median pred/ms
moves 0.7893 → **0.7473** (item 10), → **0.8440** (item 11), → **0.8649** (item 23 b, with p90 crossing 1 at
1.1413). One item-11 row is predicted 37,731.3 against ms 37,738 (pred/ms 0.99987).
(`tree-identity-saturation.md` §3.2, §3.3; `linear-tree-model-b.md` §4.1, §4.2;
`holdout-month-decides-rounds.md` §4.2, §4.3.)

Item 30's split is the mirror image of the other three: pooling **halves the non-LIRF unmatched error**
(1,034.74 → **486.57** for `P-restr`, 487.98 for `P-full`) and improves **nine of ten airports** (EDDF
1,683.29 → 503.04, EHAM 1,184.00 → 286.24, LEBL 1,150.42 → 275.02, LFPG 1,217.00 → 684.30), while **LIRF
gets worse** (8,578.38 → **9,689.57** / 9,733.56). Inside LIRF: ms < 2 h (82 rows) 1,974.3 → 1,914.2; ms ≥ 2 h
(58 rows) 13,119.3 → **14,881.0**; sched-filled \|y − ms\| ≤ 60 (99 rows) 3,597.4 → 3,400.8; not sched-filled
(41 rows) 14,833.4 → **17,107.5**. The LIRF share of unmatched SSE rises 0.771 → **0.951** and the four worst
rows' share 0.677 → 0.878. All four LIRF ms ≥ 12 h rows move **down**, away from labels that are all ≥ 47,767
(32,351.7 / 34,539.7 / 34,093.3 / 32,399.5 → 28,161.2 / 22,541.8 / 25,733.1 / 24,287.4; RMSE on those four
47,312.7 → **55,069.2**), even though 72 % of LIRF unmatched predictions move up. Trees per seed 45 → **152**
(`P-restr`) / **145** (`P-full`), because early stopping now reads a **185,674**-row October holdout instead of a
1,913-row one. Dropping the four all-null-on-unmatched categoricals is worth **−9.56 s** of unmatched RMSE
inside a pooled fit (2,154.29 → 2,144.73) and **nothing** outside one.
(`shared-b-all-rows.md` §2.5, §3.1, §3.2, §3.3, §3.5.)

### 3.3 Protocol C — post-processing the logged OOF (baseline unmatched 2,296.1, total 334.58)

| variant | unmatched OOF RMSE | delta | notes |
|---|---|---|---|
| clip unmatched at 3,600 / 7,200 / 14,400 / 43,200 / 86,400 | 3,567.6 / 3,373.4 / 3,069.7 / 2,305.8 / 2,296.1 | +1,271.5 / +1,077.3 / +773.6 / +9.7 / 0 | total 452.59 / 433.64 / 404.56 / 335.42 / 334.58; the cap-quantile curve is monotone in q |
| shrink 0.5 toward the airport median (training labels) | 2,771.6 | +475.5 | toward the OOF-label median **2,824.7** (reproduces M's 2,825); constant 3,572.4 / **3,724.2**; oracle airport-mean **3,492.0** |
| shrink 0.5 toward the (airport, hour) median | 2,767.4 (`hour_utc`) / 2,736.9 (`hour_local`) | +471.3 / +440.8 | differs from the airport target by ≤ 35 s |
| **cap non-LIRF unmatched only at 3,600** | **2,207.3** | **−88.8** (total 334.58 → **327.05**) | 53 rows affected, 49 helped / 4 hurt; SSE delta negative in **each of the 4 OOF months** and at **all 8** affected airports; best of the sweep is 1,800–2,400 → **2,201.5** (total 326.57) with 147 / 42 rows hurt; 7,200 → 2,234.3 |
| cap LIRF unmatched only at 3,600 / 7,200 / 14,400 | 3,623.2 / 3,414.6 / 3,075.9 | +1,327 / +1,119 / +780 | the whole deterioration of the unrestricted cap is the LIRF part |
| **pred := ms on LIRF unmatched, ms ≥ 2 h** (no flag) | **1,892.7** | −403.4 | reproduced identically by `rule-overlays-extreme-rows.md` at T = 7,200 (total 301.487) |
| **pred := ms on LIRF unmatched, ms ≥ 14,400** (rule S-msU) | **1,752.57** | −543.5 (total 334.58 → **290.739**, −43.84) | 55 rows, 51 improved / 4 worsened; **69.6 %** of the SSE reduction comes from **7 rollover rows**; removing their contribution by arithmetic leaves total ≈ 321.9 (−12.7) |
| oracle "f10 rows → ms" / "r1800 rows → ms" | 2,173.8 / 2,062.3 | −122.3 / −233.8 | the flag is `BLOCK_TIME`-derived and null on every ranking row |
| replace non-LIRF unmatched ms ≥ 4 h by the airport median | 2,217.2 | −78.9 (total 327.88) | 164 rows; oracle bound 2,214.7 (total 327.68), i.e. the median recovers 97 % of the oracle reduction; 87 % of the gain is the 21 ms ≥ 12 h rows |
| leave-one-month-out lookup mean(y \| ADEP × mms-bin) | **1,755** vs 2,296 | −541 | **plan-carried figure (M), not re-derived in `results/`** (`mms-mixture-reparam.md` §1 states the plan's no-training lookup and oracle figures were not re-derived) |

(`tail-postprocessing-cap-shrink.md` §3.2–§3.6; `sched-filled-block.md` §J; `rule-overlays-extreme-rows.md` §D,
§F; `false-alarm-non-lirf.md` §D; `plan.md` §1 #9 for the LOMO lookup.)

### 3.4 Baseline caveats, restated so the numbers are not mixed

- **2,326.5** is the three-fold reproduction in the item-9 / item-14 script (num_threads 2); **2,296.1** is
  `oof_main-48c19bc830` (num_threads 4). Every Protocol-A delta is against 2,326.5.
- **2,109.32** is the unmatched RMSE of the forward fold **alone** (3,005 rows), the baseline of Protocol B.
  The same fold reads 2,109.3 in the OOF file, **2,177.4** in the item-9 script's reproduction, and
  **2,109.3167** in item 30's, which is bit-exact against the logged record (row-by-row max \|diff\| 0.0000).
  `shared-b-all-rows.md` §6 I4 records the code difference behind the gap — the item-9 script adds its own `ym`
  fold key with `with_columns` and does not remove it from `visible_numeric_columns`, so model B there was given
  one extra feature and a differently ordered matrix (which `feature_fraction: 0.8` samples by index), whereas
  item 30's script drops `ym` and preserves the parquet's column order. The code difference is a fact; that it
  **causes** the +68.1 s gap is that report's own inference.
- Protocol A's tail-trimmed CIs are `robustness_curve` on the **unmatched rows only**. Protocol B's are
  `make compare` on **all 328,009 rows**, where the unmatched rows are 0.9 % of the population — the two K = 200
  columns are not the same statistic.
- `rounds_unmatched` differs across protocols: 45 (forward fold), 566 / 288 / 45 (Protocol A's three folds),
  199–658 (production, quoted from P / M / H and **not** re-measured in `results/`).

---

## 4. Contradictions between reports, and how they were settled

1. **Items 13 vs 14 (post-processing helps vs robust losses harm).** `plan.md` §1 kept them as separate,
   conflicting items. **Settled — they are not in conflict.** Both find the same split: model B's large **LIRF**
   predictions have large labels and its large **non-LIRF** predictions do not. Item 13: capping *all* unmatched
   predictions worsens RMSE (2,296.1 → 3,373.4 at 7,200) while capping *non-LIRF only* improves it
   (2,296.1 → 2,207.3 at 3,600; total 334.58 → 327.05). Item 14: every robust loss improves non-LIRF
   (1,451.7 → 1,271–1,360) and worsens LIRF (7,326.3 → 8,339–13,932). A loss change cannot separate the two
   because it acts on all rows at once.
   — `robust-loss-harmful.md` §4 ("Item 14 vs item 13 … RECONCILED"); `tail-postprocessing-cap-shrink.md` §4 (d)

2. **The two `shared_b` sides (item 30 side A vs item 31 side B).** **Now settled by measurement — the two sides
   are each right about a different part of the data.** The pre-check had already settled the data facts: B1–B4
   CONFIRMED (matched f120 0.114–0.257; matched y ≥ 2 h = 103 of 2,062,577; 119 of 122 y ≥ 4 h rows unmatched;
   0 actype-null matched rows), A2 REFUTED (33 → 34, not ≈ 2,000), A1 split (confirmed for airport / hour /
   runway / stand / traffic / weather columns, refuted for aircraft and operator columns), B5 qualified (32
   matched LIRF rows do carry the fill mechanism) — and left **A4** and **B6** UNDETERMINED pending the run
   (`shared-b-no-transfer.md` §4, §5). The forward-fold run has now been done:

   - **A4 ("pooling improves unmatched predictions") — REFUTED as stated.** Unmatched RMSE **2,109.32 → 2,144.73**
     (restricted columns) / **2,154.29** (the literal `shared_b`); whole fold **284.7038 → 287.1174 / 287.7726**.
     The full-data difference is **+35.41 [−409.48, +275.50]** / **+44.98 [−397.22, +284.72]** — a non-significant
     *worsening*, not an improvement — and `prc.compare.verdict` returns **REJECT** ("no difference on all rows")
     for both.
   - **B6 ("pooling will not help the tail") — CONFIRMED.** LIRF unmatched **8,578.38 → 9,689.57 / 9,733.56**;
     LIRF ms ≥ 2 h 13,119.3 → **14,881.0**; LIRF not-sched-filled (41 rows) 14,833.4 → **17,107.5**; all four LIRF
     ms ≥ 12 h predictions move down and away from their labels (RMSE 47,312.7 → **55,069.2**).
   - **A4b (side A's mechanism, off the tail) — CONFIRMED.** Non-LIRF unmatched **1,034.74 → 486.57**, better at
     nine of ten airports, and the gain survives the robustness checks: **K = 200 −25.37 [−36.36, −13.11]** and
     **K = 2000 −11.61 [−15.19, −8.40]** on the unmatched-only curve, both CIs excluding zero — recorded by the
     report as the only such result among its variants.
   - **B5′ (side B's named mechanism) — CONFIRMED, consistent with the pre-check.** No tree of either pooled
     model splits on `nm_matched` / `eobt1_null` / `lobt_null` at the **root** (earliest at depth 2; 171 / 113
     splits in 78 of 152 / 61 of 145 trees; 1.72 % / 1.57 % of total split gain), so the unmatched rows share
     leaves with matched rows above depth 2 in most trees.

   The report's own summary: "the measurement supports side B on the question that decides the score, and side A
   only on the part of the data that does not decide it." Its I1 marks the shared-leaf reading as **inference**:
   the run measures the two outcomes and the absence of a root-level regime split, not the causal path.
   — `shared-b-all-rows.md` §3.1–§3.5, §4, §6; `shared-b-no-transfer.md` §4, §5

3. **"top-20 rows carry 73.2 %, all 20 LIRF" (P, `plan.md` §1 #3) vs "80.4 %, 18 LIRF + 2 EGLL".**
   **Settled by re-derivation.** `lirf-regime-tail-count.md` §3.4 joins the submission files to
   `submitting.parquet` on `MVT_ID_mvt`, lists all 20 rows, and reports **80.4 %** with **18 LIRF + 2 EGLL**
   (the EGLL rows have ms 25,438 / 13,199 and v4 100 / 1,603 against v8 12,299 / 12,700); its §4 records the
   non-reproduction of P's figure explicitly. Both numbers are stated here; the 80.4 % figure is the one with a
   row listing behind it, and `refit-seed-semantics-changed.md` §3.2 cites it as the reference scale.
   *Related and already settled:* RMS(v8 − v4) on unmatched = **1735.5** (P) vs **1,736** (M) is the same
   quantity at different rounding (`plan.md` §4-7); the re-derivation gives 1,735.5.

4. **SSE shares for the rollover rows (`plan.md` §4-4: D 34.7 %, M 52.7 % of unmatched ≈ 25 % of total,
   H 24.1 %).** **Settled — three different row sets and two different OOF files.** On `oof_main-48c19bc830`:
   D's 10 rows with y > 20 h are **0.3472 of total SSE** (RMSE 51,121, pred median 37,131) — confirmed. M's
   8-row set is identified as the 9 unmatched y > 20 h rows **minus NOSOS610** (whose \|y − ms\| = 1,503 < 1,800
   classes it sched-filled under M's rule): those 8 are **0.5267 of unmatched SSE** (RMSE 53,927) — confirmed —
   but **30.9 %, not ≈ 25 %, of total SSE**; 25.5 % is the share of the 9 rows with y ≥ 86,400. H's **24.1 %**
   reproduces exactly on **`oof_main-f9d57a661f`** (total 300.17, unmatched 1,864.9), where the same 10 rows are
   24.1 %; across all 672,428-row OOF files they range **23.7 %–39.9 %** of total SSE and 50.1 %–62.2 % of
   unmatched SSE. — `block-day-rollover.md` §3.5, §3.6, §4

5. **Unmatched definitions (routing 22,470 / 5,290 vs FLIGHT_ID 22,424 / 5,283).** **Settled.** Every Batch A
   report computed both. The difference is 46 training / 7 ranking rows, none of them LIRF tail rows; no share
   moves by more than 0.05 pp, no RMSE by more than 3 s, and every LIRF-related quantity is identical.
   `AOBT_3_flt` null ≡ `mvt_minus_aobt3` null on every row of both frames, so the two wordings of the routing
   rule are one rule. — `sched-filled-block.md` §B; `unmatched-composition-shift.md` §3;
   `fold-shape-vs-ranking.md` §3.4

6. **Sched-filled fraction at LIRF given ms ≥ 2 h (D 64 % on 622 rows / 63 % on 627; M 0.633).** **Settled.**
   Both are right on their own row set: the 2–20 h ms band has **622** rows with f10 **0.635**; ms ≥ 2 h with no
   upper bound has **627** rows with f10 **0.6316**. The 622 vs 627 gap is the > 20 h band. M's "≤ 0.004
   elsewhere" holds under the BLOCK − SCHED definitions (non-LIRF f10 0.0002, f120 0.0005) but **not** under M's
   own \|y − ms\| < 1,800 definition (0.0081 pooled, 0.0677 at LTFM), which admits ordinary long delays.
   — `sched-filled-block.md` §G, §4

7. **"632 sched-filled rows carry 10.5 % of unmatched SSE" (M) vs D's ≤ 10 s fill.** **Settled by threshold:**
   632 rows is `f120` (share 0.1052); `f10` gives 353 rows (0.1036); `r1800` gives 2,348 rows (0.1933). The
   BLOCK-based definitions agree on the SSE share because the extra 279 rows have small errors.
   — `sched-filled-block.md` §J, §4

8. **Residual quantile on unmatched y ≥ 4 h rows: M's 0.9 quantile 20,922 vs 20,706.** **Settled —
   quantile interpolation** (the two nearest order statistics are 20,706 and 21,784). — `sched-filled-block.md` §E

9. **Predictions > 12 h per submission: P's 2 / 16 / 18 / 2 / 2 / 2 vs D's "3–19".** **Settled by row set:**
   P counted the **5,290 unmatched** rows, D counted **all 344,841** rows. Both are reproduced: unmatched
   2 / 16 / 18 / 2 / 2 / 2 and all-rows 3 / 17 / 19 / 3 / 3 / 3 — the difference is the single matched prediction
   of ≈ 84.8 k present in every file (v7 84,866, v9 84,814), the ranking analogue of the matched rollover row
   AEZ146R. — `lirf-regime-tail-count.md` §3.4; `tail-postprocessing-cap-shrink.md` §3.9

10. **D's "22 ranking matched rows predicted ≥ 2 h = EGLL 7, LIRF 7, EHAM 6, LFPG 2".** **Settled by
    definition:** that exact split reproduces only as **matched(FLIGHT_ID) ∧ ms ≥ 2 h** (22 of 23). Under the
    routing definition without the ms filter the count is also 22, but **LIRF 8 / EGLL 7 / EHAM 6 / LFPG 1**.
    The two differ by one AOBT_3-null LFPG row (routed to model B, pred 10,899) and one LIRF row with ms 6,837.
    — `matched-proxy-outliers.md` §3.3, §4 (7)

11. **LFPG unmatched OOF cell: 1,565 rows (routing) vs D's 1,562 (FLIGHT_ID).** **Settled** — definitional; the
    SSE share is identical at 15.48 %. — `lirf-matched-fill.md` §4 (F3)

12. **Inner-holdout extreme count (`plan.md` §4-3: H "0–5", P "6 with y ≥ 14,400", M "December 6 ≥ 4 h, max
    87,605").** **Partly settled.** The holdout **size** is reproduced at 1,649 December rows by
    `holdout-encoding-self-inclusion.md` §2, and October's composition is reproduced exactly by
    `holdout-month-decides-rounds.md` §4.1 (1,913 rows, **4** ≥ 4 h, **0** ≥ 12 h, max y 25,558 — M's figures).
    **December's extreme count was not re-measured**: `holdout-month-decides-rounds.md` §6 records that
    reproducing production's December-holdout figures (199–658 rounds, 1,649 rows, 6 ≥ 4 h) needs a full
    12-month fit, which was out of scope. So P's "6" stands unverified against H's "0–5".

13. **Which file the leaderboard scored as v2 (312.02).** **Open.** The repo `outputs/v2_two_clock.parquet` and
    the bucket copy `uploaded/v2.parquet` differ on 341,701 rows, RMS **227.4** overall and **1,766.9 on
    unmatched rows**; the bucket copy is within **260.8** s RMS of bucket v3. `submissions.jsonl` names the repo
    path, the old-code worktree rebuild equals the repo file **exactly on all 5,290 unmatched rows**, and
    `refit-seed-semantics-changed.md` §3.1 reproduces the repo file exactly at 658 rounds with old seed
    semantics. The bucket copy's provenance is not recorded. `false-alarm-non-lirf.md` reports both wherever v2
    appears and leaves the identity open (§5).

14. **Whether `outputs/v5_unmatched_capacity.parquet` is the uploaded v5.** **Open.** P's v5 row matches this
    file exactly (1,060 / 1,559 / 9,693 / 61,356 / 106 / 32 / 2), but the file is dated 09-14 02:04 while the
    uploaded v5 was `outputs/main.parquet` at 09:54, since overwritten. Every "v5" statement in
    `false-alarm-non-lirf.md` and `lirf-regime-tail-count.md` is about `v5cand`.
    — `lirf-regime-tail-count.md` §5; `false-alarm-non-lirf.md` §5

15. **Which pair `docs/handoff.md` §5 measured for "62 s / 0.77 s".** **Open** — not identified in the handoff,
    and no available pair reproduces it. — `decision-rule-blind-to-tail.md` §5

16. **M's leave-one-month-out "bin-mean 44,405" on the forward fold.** **Open.** The other three figures of that
    line reproduce exactly (n = 10, mean y **48,907**, mean OOF **26,296**, mean ms **42,960**), but 44,405 is
    not reproduced under either binning (29,473 with a single bin ms ≥ 4 h; 19,466 / 66,298 with the plan's two
    bins, row-weighted 38,199). M's bin edges are not stated in the plan.
    — `cond-mean-vs-median.md` §3.5, §4 (E)

17. **Whether the plan's `compare.py` / `cv.py` line numbers refer to some earlier version.** **Open.** They do
    not exist at HEAD, and `git` history shows `compare.py` was created at 170 lines and no revision reached
    line 470. — `decision-rule-blind-to-tail.md` §3.1, §5

18. **Which OOF file backs `docs/README.md` §8's v2 → v6 row.** **Open** — the v2-era OOF is not in
    `experiments/`. — `decision-rule-blind-to-tail.md` §3.6, §5

---

## 5. Still open

### 5.1 What no local measurement can decide — it needs the 2026 labels

`TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt` are null on **all 344,841** ranking rows
(`sched-filled-block.md` §K; `block-day-rollover.md` §3.7; `matched-proxy-outliers.md` §3.3). Consequently:

- **The leaderboard effect of every variant in §3.** Every result file states this in its own §5.
- **Whether the ranking LIRF rows are sched-filled at the training rate.** The density match is established
  (383 vs 397 rows; 181 vs 181 with ms ≥ 2 h) but the fill rate is not observable. The two source agents suppose
  opposite things about the ~11–20 rows: **P supposes y ≈ 86,400 (rollover), M supposes y = ms with training
  p = 0.62**. — `lirf-regime-tail-count.md` §5; `sched-filled-block.md` §5; `block-day-rollover.md` §5
- **How many rollover rows the ranking set contains.** Only `ms` is observable; the candidates by ms alone are
  the 11 LIRF unmatched rows with ms ≥ 12 h and the 6 + 1 unmatched / 36 + 6 matched rows in the 20–28 h /
  > 28 h ms bands. — `block-day-rollover.md` §5
- **Which unmatched rows carry the 8.373e9 unmatched-side v4 → v8 LB gap.** Only the total is fixed by the v9
  identity; LB gap / local Σgap² = **0.53**, so the disagreement rows are not uniformly "v4 right, v8 wrong".
  — `lirf-regime-tail-count.md` §4 (C), §5
- **What produced v4's 18 predictions > 43,200.** Candidates left standing after item 22: `linear_tree` on
  model B, the v4-era METAR cache and weather join, the 1,500-round cap, the absence of `force_row_wise`, the
  library versions of 2026-09-10 — **none reproducible from this repository**.
  — `hour-utc-absent-in-v3v4.md` §5; `v4-config-not-reproducible.md` §4, §5
- **Whether the 595 EHAM Jan 2026 disruption-day unmatched rows carry ordinary labels or an artefact**, and the
  operational cause of the event. Bound only: at RMSE 1,000 / 2,000 / 3,000 / 5,000 those rows contribute
  1,725 / 6,902 / 15,529 / 43,136 s² to overall MSE. — `eham-jan2026-disruption.md` §4 (G), §5
- **Whether the 280 non-LIRF unmatched ms ≥ 4 h ranking rows have training-like labels.** Table F of
  `false-alarm-non-lirf.md` is explicitly conditional on that; item 15's EHAM excess is a reason it may not hold.
- **Whether the ranking LIRF matched rows are filled at the training rate** (expected 4,725–4,884 filled rows
  under training rates, of which ≈ 1.1 k filled ∧ far). — `lirf-matched-fill.md` §5, I3
- **Whether the 16 ranking matched twins carry inflated labels.** BLOCK is null, so the 98/98 BLOCK-ordering test
  cannot be repeated. — `return-to-stand-twins.md` §5
- **Whether the per-cell error in 2026 equals the per-cell OOF error in 2025** — the assumption behind both
  reweighting arithmetics (unmatched +9.6 %, matched +7.0 s). — `unmatched-composition-shift.md` §5;
  `sched-anchored-delay-shift.md` §5
- **Which timestamp carries the wrong date on the 15 unmatched rollover rows.** No single one-day shift makes
  SCHED / BLOCK / MVT consistent for the 12 LIRF rows; for the only row with an independent clock the evidence
  points at MVT, not BLOCK. — `block-day-rollover.md` §4, §5

### 5.2 What needs a run nobody has done

- **Item 30 `shared-b-all-rows` — now run** (forward fold only, four fits, 476.7 s wall), so items 30 and 31 are
  settled as far as a local fold can settle them (§4 item 2). What that run explicitly leaves open:
  the **other two folds** (`jan` / `jul` — only `forward` was run, and its month-level bootstrap has 2 months,
  giving [−79.80, +143.45]); **pooling combined with item 9's variant A**, which moves the tail while pooling
  moves the body — "whether they compose is untested"; **hyper-parameters chosen for a 1.76 M-row model B**
  (127 leaves / min_data 50 / early stopping on a 185,674-row holdout are the current config's numbers, and
  `train.py::_fit` passes no sample weights, so a regime-stratified inner holdout or an unmatched-row weight is a
  different experiment); and **whether `AIRCRAFT_TYPE_mvt` learned on matched rows helps or hurts the LIRF rows**,
  where it is null on 1,487 of 1,488 training rows although it is the most frequent root split of both pooled
  models. — `shared-b-all-rows.md` §5
- **Items 32, 33, 34, 35** (`arrival-turnaround-features`, `peer-following-context-features`,
  `airport-specialists-blend`, `explicit-clock-difference-features`) — all four are Batch C full runs
  (~20 min each) and **none was run**; no result file exists for them, and their no-training pre-checks are
  unexecuted except where a sibling item covers them incidentally. — `plan.md` §3 Batch C
- **Isolating `linear_tree` on model B alone** requires a code change (an `unmatched_use_linear_tree` field),
  which this investigation was not permitted to make. — `linear-tree-model-b.md` §6
- **A robust loss applied to non-LIRF rows only** — a per-row-group objective is not expressible through
  `configs/main.yaml`. — `robust-loss-harmful.md` §5
- **A byte-exact four-thread rerun of the pipeline**, so the model-B variants can be fitted against the logged
  baseline instead of a 2,326.5 reproduction. — `mms-mixture-reparam.md` §5; `robust-loss-harmful.md` §5
- **A full 12-month fit** to reproduce production's December holdout and its 199–658 rounds.
  — `holdout-month-decides-rounds.md` §6
- **Sweeps not traced:** `unmatched_min_data_in_leaf` beyond {50, 5}; `linear_lambda` ≠ 1.0; linear_tree at a
  different leaf size; huber alphas between 0.9 and 5,000; `regression_l1` (not fitted separately); 3-month or
  random (non-calendar) inner holdouts. — `tree-identity-saturation.md` §5; `linear-tree-model-b.md` §6;
  `robust-loss-harmful.md` §5; `holdout-month-decides-rounds.md` §6
- **A controlled weather on/off pair at one sha.** The only existing pair (`det_a` → `weather_check`) spans four
  commits including `force_row_wise`; the `prc.compare` 200-row curve quoted at `docs/README.md` line 89 was not
  re-run. — `seasonal-deicing-shift.md` §5
- **The encoding-leak experiment on model A** (2.06 M rows, ~170 k holdout) — flagged expensive by P, not run.
  — `holdout-encoding-self-inclusion.md` §5
- **Separating "leak removed" from "training data 11 → 6 months"** in `jul` 375.01 vs `jul_fwd` 421.69 — needs a
  control run (e.g. `jul` restricted to 6 later months). `docs/README.md` lines 119–121 state the confound.
  — `fold-shape-vs-ranking.md` §5
- **Whether a different adopt/reject rule would have produced a better LB.** The flip table exists
  (`decision-rule-blind-to-tail.md` §3.6) but **no rule's LB outcome is measured**.
- **Whether `dest_distance_km` in the current cache is populated** (the `except Exception` fallback would cache a
  null column with an identical fingerprint) — needs a frame load. — `cache-fingerprint-gaps.md` §5
- **Whether `window_coverage_60m` holds only {0, 1}** — needs row reads; recorded as UNDETERMINED.
  — `n-features-72-mislabel.md` §4 (j)
- **Whether the IEM `p01i` field is genuinely empty or lost in `parse_metar_csv`** — needs the raw CSVs under
  `data/ref/metar/`. — `seasonal-deicing-shift.md` §5
- **Whether the trained GBDTs already exploit `mvt_minus_sched mod 60` or `ADEP_mvt`** to the same effect as an
  explicit second-of-minute indicator — needs a fit. — `timestamp-precision-clusters.md` §5
- **Whether a trained classifier recovers the LIRF matched fill state** better than the group-mean R² ≤ 0.238
  indicates — needs a fit. — `lirf-matched-fill.md` §5
- **Whether an ADEP × ms interaction fixes the non-LIRF false alarms inside the GBDT** — recorded as NOT TESTED,
  requires a fit. — `false-alarm-non-lirf.md` §4 (last row)

### 5.3 What needs a probe submission

`plan.md` §3 **Batch D is NOT scheduled**. Two things sit in it and neither was run:

- **Item 29 `v5-verdict-confounded`** — a same-code 255/20 vs 127/50 comparison via submission, the only way to
  settle R10 above.
- **Item 11 step (c)** — a probe of v4-matched + candidate-unmatched, the only way to test the sentence
  "v4's 37 s LB advantage is mostly `linear_tree` on ~2 ranking rows", which `linear-tree-model-b.md` §5 records
  as **UNDETERMINED**.
- **Item 30's LB effect** — `shared-b-all-rows.md` §5 names "a probe submission (v4-matched + pooled-unmatched)"
  as "the only local measurement of it", and records that it **was not run**. The ranking set has 383 LIRF
  unmatched rows (181 with mms ≥ 2 h, 11 with mms ≥ 12 h) whose true labels decide whether moving the LIRF tail
  predictions down costs or earns seconds.

**A probe submission requires the user's explicit authorisation, and none has been given.** `plan.md` §3 marks
Batch D "NOT scheduled — needs the user's explicit authorization"; `docs/README.md` line 253 states
「提出は依頼があるまで行わない」. The recorded constraints are 5 submissions per day and 1 GB
(`docs/README.md` §1, `docs/handoff.md` §1), and `plan.md` §3 quotes ranking.html: "We will monitor submissions
for attempts to learn from or exploit the ranking process. We consider such practices unfair and inconsistent with the purposes and goals of the Data Challenge."

### 5.4 Provenance questions left open

Listed in §4 items 13–18: which file was scored as v2; whether `v5_unmatched_capacity.parquet` is the uploaded
v5; which pair gave "62 s / 0.77 s"; M's LOMO bin edges; the plan's `compare.py` line numbers; which OOF file
backs `docs/README.md` §8's v2 → v6 row. Also: **v3 / v4 have no run record in any commit**, so the exact
`features.py`, weather cache, round counts and library versions behind LB 305.03 are not recoverable
(`v4-config-not-reproducible.md` §3.1; `cache-fingerprint-gaps.md` §5; `n-features-72-mislabel.md` §5), and
`data/work/adv_score.parquet` has **no producer in the repository** (`src/prc/adversarial.py::main()` writes no
file) — `sched-anchored-delay-shift.md` §3.6.

---

## 6. Structural facts about the code that surfaced

These are facts about the repository, not findings about the data.

**Model configuration surface**

1. **`use_linear_tree` is a single shared `TrainConfig` field** (train.py:80). `lgb_params()` (129–135) adds
   `linear_tree` / `linear_lambda` whenever it is true. `for_unmatched()` (117–127) overrides **only**
   `num_leaves`, `min_data_in_leaf` and `num_rounds`, so the flag is inherited unchanged; `predict_fold` calls
   `_fit(..., cfg, ...)` for model A (333) and `_fit(..., cfg.for_unmatched(), ...)` for model B (358–359).
   There is no `unmatched_use_linear_tree` field and `pipeline.py` copies only dataclass fields, so **no YAML
   key can separate the two models**. — `linear-tree-model-b.md` §2

2. **Under today's config `for_unmatched()` returns the same parameters as `cfg`.** `configs/main.yaml` sets
   `unmatched_num_leaves: 127`, `unmatched_min_data_in_leaf: 50`, `unmatched_num_rounds: 2000`, which are the main
   model's values, so `for_unmatched()` resolves to (127, 50, 2000) — identical to `cfg` on the three fields it
   overrides. The one design decision of the removed `shared_b` mode (5b90919: "capacity follows the training
   set", i.e. `b_cfg = cfg` rather than `cfg.for_unmatched()`) is therefore **not a difference** as the repository
   stands; it would have been one under the deleted `configs/p1_shared_b.yaml`, which set 255 / 20 / 3000 for the
   unmatched model. — `shared-b-all-rows.md` §2.1, §2.2, §4 (A-cap)

3. **Model B's round count cannot be fixed from a config, and its early stopping cannot be switched off.**
   `_fit` (252–300) runs early stopping whenever `0 < n_hold < train.height`, then sets
   `rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))` —
   `num_rounds` is a **cap and a scaling target, never a fixed count**. `inner_holdout_months` and
   `early_stopping` are inherited from the shared config (there is no `unmatched_inner_holdout_months` or
   `unmatched_early_stopping`), so `inner_holdout_months: 0` would disable early stopping for model A as well.
   Empirically `unmatched_num_rounds: 600` gives a **bit-identical** run.
   — `holdout-month-decides-rounds.md` §2, §4.2

4. **The inner holdout is per model and is the latest calendar month of that fold's training side.**
   `_inner_holdout_mask` (239–250) takes `uniq[-months:]` and is applied inside `_fit`, separately to the
   matched and unmatched frames — model B's forward-fold holdout is October's 1,913 unmatched rows while model
   A's is October's 183,761 matched rows. — `holdout-month-decides-rounds.md` §4.1

5. **Label encodings are fitted on the whole training side including the holdout month.** `predict_fold`
   (317–318) fits ref / arh / apt / op on all labelled training rows and `_fit` then carves the holdout month
   out of the same frame, so every holdout row's `ref_*` / `arh_med` / `apt_med` contains its own label
   (December contributes ~8–10 % of each group at the median, 100 % at the maximum; 4.1 % of holdout rows sit in
   a ref group more than 20 % December). The **only channel** from the leak to the shipped model is the round
   count, because `_fit` uses the early-stop booster for nothing but `best_iteration`.
   — `holdout-encoding-self-inclusion.md` §3.1, §6

6. **`cols_b` is a prefix-exclusion list.** Model B gets the numeric columns not prefixed by `mvt_minus_aobt3`,
   `aobt3`, `lobt_minus`, `mvt_minus_lobt`, `mvt_minus_eobt1`, `mvt_minus_iobt` (train.py 350–356), plus the 10
   `CATEGORICALS`. `mvt_minus_sched` and `mvt_minus_sched_slog` carry none of those prefixes and
   `configs/main.yaml` has no `exclude_features` key, so both are given to model B; ranking null rate 0.
   — `sched-filled-block.md` §Code

7. **Three numeric columns inside `cols_b` separate the two regimes exactly with one split each.**
   `nm_matched` is 1 on all 2,062,577 matched rows and 0 on 22,424 of 22,470 unmatched rows (1 on the 46
   routing-only rows); `eobt1_null` and `lobt_null` are 0 on **every** matched row and 1 on **every** unmatched
   row. The four `_flt` categoricals do the same through `__NA__`. `_fit` passes **no sample weights** (the word
   "weight" does not occur in the file). The removed `shared_b` mode (git 5b90919, removed in ea04ec9) fitted
   `u_tr = tr` (all rows) with `cols_b` unchanged and the **main-model** capacity (`cfg`, not
   `cfg.for_unmatched()`). — `shared-b-no-transfer.md` §G, §G2

8. **`frozen_wx` and `frozen_wx_prev1h` are Boolean** and are therefore excluded by `numeric_feature_columns`,
   so weather contributes **18** numeric columns, not 20. — `n-features-72-mislabel.md` §3.2;
   `cache-fingerprint-gaps.md` §2

9. **`deicing_gate`'s precipitation term is inert.** `precip_mm > 0` occurs in **0 of 138,446** METAR
   station-hours, so `deicing_gate ≡ frozen_wx ∧ temp_c ≤ 3` on every row (1,326 gate hours, 0 disagreements)
   and `temp_x_precip` is identically 0. — `seasonal-deicing-shift.md` §3.1; `eham-jan2026-disruption.md` §3.6

10. **There is no second-of-minute feature.** No `.dt.second`, `% 60`, `mod(60`, `sec0` or `second_zero` anywhere
   in `src/prc/`; `minute_of_day_utc` is `hour·60 + minute`. It is algebraically present, though: SCHED seconds
   are 0 on every row, so `mvt_minus_sched mod 60 == 0 ⇔ MVT second == 0` on all 2,085,047 rows, and
   `mvt_minus_aobt3 mod 60 == 0 ⇔ sec(MVT) == sec(AOBT_3)` on all 2,062,577 matched rows.
   — `timestamp-precision-clusters.md` §3.7

**Caching, logging and provenance**

11. **The feature-cache fingerprint covers only `features.py`.** `cache_fingerprint` (pipeline.py 55–71,
    unchanged since c2cd247) = `raw_dir`, sorted (name, byte size) of every raw parquet, `train_rows`,
    `rank_rows`, `use_weather`, and `sha256(src/prc/features.py)[:16]` of the **working-tree** file. Not in the
    key: `weather.py`, `airports.py`, `schema.py`, `data/ref/metar/*.csv`, `data/ref/metar_hourly.parquet`,
    `data/external/airports.parquet`, `train.py`, `pipeline.py`, library versions. Before c2cd247 the cache was
    keyed by **path alone**. Because `use_weather` is in the key, alternating weather-on/weather-off configs
    rebuild the cache every time (cost, not staleness). — `cache-fingerprint-gaps.md` §3.1, §4

12. **`git_sha` is captured at the end of CV, not the start.** It is evaluated in `run_cv` after the fold loop
    (train.py:448), so it is HEAD at the **end** of the run. **17 of the 35** current records have their
    `git_sha` committed 31–999 s after the estimated start; for `features.py` this never mattered, but for
    `train.py` / `pipeline.py` content the recorded sha may post-date the code that ran (one record has three
    `train.py` commits inside its window). `experiments.jsonl` carries **no timestamp** and no `feature_code`.
    — `cache-fingerprint-gaps.md` §3.3, §4

13. **`overall` silently drops `jul_fwd`.** `run_cv` (train.py 427–444) starts `oof` all-null and left-joins each
    fold's predictions with `pl.coalesce([oof, _p])` in fold order `[jan, jul, jul_fwd, forward]`. `jul` and
    `jul_fwd` predict exactly the same 190,713 rows, so **`jul_fwd`'s predictions are never written**:
    `overall.n = 672,428` while `Σ fold n = 863,141`. Verified in all six four-fold records (`overall.rmse_full`
    = pooled{jan, jul, forward} to 3 dp, differing from pooled{jan, jul_fwd, forward} by 7.6–14.0 s) and by a
    toy replay. `configs/main.yaml` and `configs/submitted_v4.yaml` set no `folds:` key, so production CV runs
    all four and reports this `overall`. — `fold-shape-vs-ranking.md` §3.3

14. **`n_features` in `experiments.jsonl` has never been the matrix width.** From e0099d7 through 97e67c6 it was
    `len(numeric_feature_columns(feats))` (frame width, including `month`, excluding the 9 encodings). It changed
    at **fd848bf** to `len(visible_numeric_columns(feats, cfg))` plus `n_frame_columns`, and **no record with the
    new semantics exists** (`grep -c n_frame_columns experiments/experiments.jsonl` = 0).
    — `n-features-72-mislabel.md` §3.3

**Decision machinery**

15. **`verdict()` reads two rows of the curve and ignores the month bootstrap.**
    `verdict(curve, sign_check_level=200)` (compare.py 91–119) reads only the `dropped == 0` and
    `dropped == 200` entries; its signature has no month argument; `main` computes and prints
    `month_bootstrap_rmse_diff` (157–161) and calls `verdict(curve)` (169) without it.
    `DROP_LEVELS = (0, 20, 200, 2000)`, `SIGN_CHECK_LEVEL = 200`. As coded: REJECT "worse" if the full CI lo > 0;
    REJECT "no difference" if the full CI hi ≥ 0; ADOPT if no K = 200 entry exists; REJECT "tail-only" if the
    K = 200 CI hi ≥ 0; else ADOPT. Boundary: a full hi of exactly 0.0 yields "no difference"; a K = 200 hi of
    exactly 0.0 yields "tail-only". — `decision-rule-blind-to-tail.md` §3.1

16. **`robustness_curve` drops by `max(se_a, se_b)`** (compare.py line 43) — the rows where the two models
    disagree most — and reuses the **same seed (0) at every K**. — `decision-rule-blind-to-tail.md` §3.1, §3.4

17. **`load_pair` restricts silently.** It inner-joins on `MVT_ID_mvt` and keeps only rows where both
    predictions are non-null (compare.py 128–129); `main` prints `rows = …` (146) but **no warning** about the
    restriction. A 3-fold OOF compared with a `folds: [forward]` OOF therefore evaluates 328,009 Nov–Dec rows; a
    Jan-only vs forward-only pair evaluates **0** rows and returns "no full-data comparison"; a single-month pair
    gets a **NaN** month CI (the function returns NaN below 2 months).
    — `decision-rule-blind-to-tail.md` §3.5

18. **The routing rule is one rule in one place.** `train.py` 329 / 348 in `predict_fold`, used by both `run_cv`
    and `predict_ranking`; `AOBT_3_flt` null ≡ `mvt_minus_aobt3` null on **every row of both frames**
    (0 disagreements), and `proxy_ok_but_unmatched` (FLIGHT_ID null but proxy present) = 0 in both sets.
    — `fold-shape-vs-ranking.md` §3.4

19. **Every OOF file has 2,085,047 rows**, one per training row, with `oof` null on unscored rows; non-null
    counts are 672,428 (3-fold), 328,009 (`folds: [forward]`), 190,713 (`[jul, jul_fwd]`), 153,706 (`[jan]`).
    Aggregates that do not filter nulls report inflated `n` while RMSE / SSE are unaffected (polars skips nulls).
    `plan.md` §0's "672,428 rows" is the scored subset. — `decision-rule-blind-to-tail.md` §3.1;
    `lirf-matched-fill.md` §4 (F2); `sched-filled-block.md` §6

20. **The submission transform** is `np.clip(pred, 0, None)` in `_assemble` (train.py:377) then `np.rint` in
    `write_submission`, which also gained a `reference_median` gate (d5c4535). The clip — not the rounding — is
    what makes raw-float and submission-space RMS differ: 5 of the 20 largest disagreement rows in item 21 have
    a negative raw prediction in one or both fits. — `refit-seed-semantics-changed.md` §2, §3.2;
    `v4-config-not-reproducible.md` §3.3

21. **`data/work/adv_score.parquet` has no producer in the repository.** `src/prc/adversarial.py::main()` writes
    no file; the column is taken to be the per-row P(ranking) of that module's docstring, unverifiably.
    — `sched-anchored-delay-shift.md` §3.6

22. **A silent-null path exists in the distance feature.** `add_destination_distance`'s `except Exception`
    fallback would cache a null `dest_distance_km` column with an identical fingerprint; whether the current
    cache's column is populated was not verified. — `cache-fingerprint-gaps.md` §4 (C), §5
