# Competitor research 2026-09-21 — consolidated summary

24 slots, all complete. Every figure below is attributed to the slot that
produced it; nothing here is the commander's own analysis. Open the named
result file for the source URL, file path and [S]/[X] marking.

## A. Facts established about the competition itself

| Fact | Slot |
|---|---|
| Scored set is **Jan 2026 (152,719 DEP) + Jul 2026 (192,122 DEP) = 344,841**, 44.3%/55.7% Jul-heavy | 06, 16, 24 (five independent sources; arithmetic closes to the unit) |
| The ranking set was **replaced on 2026-09-04**: July went 3 airports -> 10, usedPairs 215,876 -> 344,841, scores moved about +25 s | 09 |
| Ranking is **best RMSE across all submissions**; no private leaderboard; a regression costs nothing | 01, 14 |
| Full per-submission history is public and unauthenticated at the OpenSky REST API; the fly.dev endpoint is a best-per-team mirror | 01 |
| Official site publishes **a registration page per team** with country, type, member count | 02 |
| We are **rank 43 of 168** at 284.3584; ranks 40-53 span 4.92 s; the rank-43 cutoff has drifted 1.146 s/day | 01, 09 |
| Anti-probing rule in full: "...unfair **and inconsistent with the purposes and goals of the Data Challenge**" | 14 |
| Organiser stated 2026-09-18 that no formal rule bans reading `AOBT_3_flt` on ranking rows; merry-quicksand asked in writing and got no ruling | 21, 22 |

## B. Corrections to things this repository had recorded wrongly

| Was recorded | Actually | Slot |
|---|---|---|
| youthful-giraffe is Matt Motoki | Official team page: Indonesia, Independent, 1 member. No source ever existed | 02 |
| 2026 jubilant-vase = PRC 2025 2nd place | Different people. PRC re-randomises codenames each edition | 04 |
| NASA pushback winners all used per-airport models | The only controlled comparison has global ahead (10.7232 vs 10.7686), better at 8 of 10 airports | 05 |
| Phoenix-Ops v8 (-7.11) is a feature addition | Same 123 predictors as v7. The gain is model capacity on clipped residuals | 07, 18 |
| ATXOT = median per runway x stand group, >=10 flights | Edition 01.00 (2023-03): P10, stand individually, >=10 flights at or below P10, rolling 12 months | 19 |
| jubilant-vase 286.66 is level with us | That is the dead board. Current: 299.2862, rank 76 | 09 |
| col3name/air-data holds the rank-21 268.52 code | Its committed submissions score 599.67 and 731.15; the 268.52 file postdates the last commit | 23 |
| satam2 = 30 features, no external data, 278.01 | `main` is 348.08; 278.01 is on another branch using 225-449 fields plus METAR and OSM | 23 |
| CDG taxi-out RMSE 1.6 min is comparable to our 284 s | Original phrasing is an average error on any given day; unit not determinable | 19 |

## C. Defects found in our own code

1. **Un-cross-fitted target encoding.** `src/prc/train.py:367-368` fits four
   label-dependent tables on `train_f` and joins them back onto those same
   rows. Greedy target statistics; CatBoost paper arXiv:1706.09516 section
   3.2 proves prediction shift; sklearn ships cross-fitting as the default.
   `reference_min_count: 30` bounds it without removing it. Slot 20.
   merry-quicksand document the identical defect in their own early-stopping
   path and never wired in their repair. Slot 21.
2. **The OOF parquet cannot support a fitted correction.** Folds carry
   different `train_months` (`jan` trains on the future, `forward` does not),
   `coalesce` resolves overlaps silently in list order, there is no fold
   label in the schema, and coverage differs between experiments (672,428
   against 328,009). Slots 11, 15. Prerequisite: add a `fold` column.

## D. Measurement we already had and had not surfaced

`stand_arrival_gap` on branch `codex/feature-engineering` (commit `8bdf4f9`),
verified from `continuation_evaluation.json`:

| mode | overall delta | airports improved | EHAM | LEMD |
|---|---|---|---|---|
| post-hoc correction | -0.781 | 7/10 | +0.374 | +0.428 |
| **as a feature** | **-3.242** | **9/10** | -0.722 | -0.263 |

The sign flips at two airports purely from changing the mode of use. Slot 18.
Slot 18 also names one column we screened and dropped: `stand_arrival_taxi`,
the joined arrival's `TAXITIME_SEC_mvt` — the same column that moved OpenAir
284.0967 -> 282.0851. We ranked candidates by correction value.

## E. Where four independent slots converged

The **unmatched / missing-proxy lane**, not features and not ensembling:

| Measurement | Slot |
|---|---|
| satam2 ablation: raw target 491.329 -> +AOBT residual 486.230 (-5.1) -> **+missing-row specialist 361.735 (-124.5)** | 23 |
| javidmardanov: matched RMSE 226.85 against missing 1157.63; missing rows are 1.5% of rows and **28.4% of total SSE** | 08 |
| merry-quicksand: their entire gap to the 271-274 tier was the unmatched lane at LIRF; the other nine airports beat the competitor | 21 |
| kind-mango: the LIRF regime head is their second-largest win; `R_norm` alone took LIRF clean 565.10 -> 318.08 | 22 |

Our own figure, from `train.py:9`: unmatched 1,875.3.

## F. Settled negatives — do not spend time here

| Claim | Evidence | Slot |
|---|---|---|
| Clean or cap the **label** | drop >120 min +24.1; drop >60 min +36.3; cap 7200 s +77.6; cap 4500 s +96.1; drop 17 rows >6 h +13.7 despite -9.9 on the bulk; Phoenix-Ops label cap 434.1 vs 382.3 | 07, 09 |
| Push the tail up | gated correction -18 s locally, **+159.6 s on the board** | 09 |
| Cross-library blending closes our gap | Sourced band 0.3-1.8% over best single = 0.9-4.5 s for us. PRC 2025 rank 1 was a single LightGBM and beat the 3-library blend by 11.16. PRC 2024 rank 2's ablation puts LGBM, MLP and SAINT at +0.00% | 03, 11, 20 |
| Congestion **counts** | Every count family has a strong negative and no clean positive anywhere. NASA 2nd place (0.055 behind 1st) shipped zero congestion features. On our own data `dep_prev_1m` correlates 0.0086 | 05, 17 |
| Per-airport specialists as a general lever | Four PRC 2025 teams shipped them, none beat the global model. Phoenix-Ops measured per-airport blend weights worse than one shared weight | 04, 12 |
| Log / Box-Cox / Yeo-Johnson under an RMSE metric | Controlled 10-dataset study: GBT squared error rises; 4.2x worse on the most skewed target | 13 |
| External data as the answer | Largest quantified gain in the entire public record is OpenAir's OPDI step, about -3.4 s. PRC 2025's whole podium used none | 16 |
| Meteostat | Every European station stops 2026-03-29/30. Null on 56% of scored rows | 16 |
| OPDI flight-events as a taxi-out measurement | v0.0.2 ships OSM crossings, not the T04-T21 model. 232 of ~30,600 departures usable (0.8%) | 16 |

## G. Positives with a measured transfer to a live leaderboard

| Change | Local | Live | Slot |
|---|---|---|---|
| Physics-identity **lower bound**: `m_aobt = MVT - AOBT_3`; where `> 1800`, `pred = max(pred, m_aobt - 3600)` | -23.9 | **-20.3** (336.46 -> 316.19), 11,826 rows | 24 |
| Deep CatBoost on **residuals clipped to +/-7200**, blended 50/50 (no new features) | — | **-7.11** | 07, 18 |
| Per-airport upper bound from 2025 label support, non-LIRF only | 0 fold price | **-2.33** on 5 rows | 15 |
| LOBT-residual expert, gate = AOBT+LOBT present, alpha 0.6308 | -2.21 | **-1.97** (271.04 -> 269.08) | 08 |

Our own confirmed-but-never-shipped instance of the third row:
`tail-postprocessing-cap-shrink.md` caps non-LIRF unmatched at 3,600 for
OOF 334.58 -> 327.05, 49 rows helped and 4 hurt. Slot 15.

## H. Protocol findings

- **Noise floors, measured by others**: seed spread 10 s, cross-machine 4 s,
  paired about 5 s (slot 09). PRC 2025's published 5-seed SD is about 7% of
  the metric, and the authors' chosen model was reversed by the hidden set
  (slot 14). Our screening resolution is 0.25 s.
- **Row bootstraps are 1.5-4.6x too narrow**; merry-quicksand moved to
  date-block bootstraps. Slot 21.
- **Blum & Hardt's Ladder** (arXiv:1502.04585) is the published,
  parameter-free form of our gate: accept only if the improvement exceeds
  `s/sqrt(n)` where `s` is the SD of per-row loss differences — inputs our
  paired bootstrap already computes. Slot 14.
- **Jan/Jul month conjunction and our two-CI rule are near-orthogonal**:
  theirs partitions the calendar, ours the sample. Both of sqcm9896's
  documented live failures passed their gate; their post-mortems name
  overfitting to holdout monsters, which a drop-200 CI is built for.
  Conversely their gate rejected a -8.6 s pooled gain. Add the arm, do not
  substitute. Slot 24.
- **Anchored-offset targets**: safe iff `anchor_time - true_off_block` is
  unimodal and small relative to `y`. Anchor on an observed time
  (merry-quicksand, `AOBT_3`) and it works; anchor on a planned time
  (kind-mango, `EOBT_1`) and it fails at +35.86. Slot 22.
- **Coverage-parity gate**: kind-mango's -13.14 from dropping external
  features was out-of-support missingness at serve time (`ec_*` went 100% ->
  0% non-null at LIRF in July 2026), not leakage or noise. Their
  `check_coverage.py` is about 90 lines and portable. Slot 22.
- **Arrival mirror**: the 344,693 arrival rows in the 2026 scoring file carry
  real taxi-in labels, so 2025->2026 transfer can be measured with labels
  before any upload. kind-mango built it and never used it as a gate; their
  one attempt to turn it into features lost +0.37 s. Slot 22.

## I. Where the literature puts us

Comparable across studies is R-squared, not RMSE — the published spread is
driven by target definition and aggregation unit, not exclusions. Our
284.36 against a within-airport SD of 514.8 s is about **R-squared 0.695**,
between Lee's CLT (about 0.69) and Yuan's ZSPD (0.77), above Lordan's LEBL
(0.449) and Liu & Itoh's RJTT (0.43). No published flight-level,
unexcluded, multi-airport, future-period result exceeds 0.80. Slot 19.
