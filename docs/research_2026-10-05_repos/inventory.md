# PRC 2026 public repo inventory (sweep 2026-10-05)

Method: GitHub repo search (MCP, authenticated), code search for each top-60 team name with `NOT repo:euctrl-pru/prc_data_challenge_website_2026` (one name per query; OR together with NOT is silently broken), code search on `AOBT_3_flt` / `TAXITIME_SEC_mvt` / `MVT_TIME_UTC_mvt` / `BLOCK_TIME_UTC`, WebSearch, and the links already in `docs/`. Every reachable repo was shallow-cloned (depth 50) into `scratchpad/repos/src/`. "Commits" is the count in the depth-50 clone, so 50 or more means the history is longer; outgoing-boat's full history is 254. "Docs" is the number of files in `docs/` that name the repo.

**Coverage check.** No top-30 team other than outgoing-boat has a public repo. Code search on the names of ranks 1–60 found mentions only inside Phoenix-Ops leaderboard snapshots and the organiser site. **jolly-lobster (#4) did publish `LeeMarshall1113/prc-taxiout-2026`.** A web search snippet attributes it to that team and mentions "an evaluation-set change mid-competition". It now returns 404 and git asks for credentials, so it is private or deleted and cannot be read.

## Inventory

| # | Repo | Team (LB rank, score 10-05) | Last commit | Commits | Licence | Docs | Approach |
|---|---|---|---|---|---|---|---|
| 1 | github.com/EnioAguiar/prc-taxiout-2026 | **outgoing-boat (#31, 243.95)** | 10-03 (work closed) | 254 | GPL-3.0 | 1 (one line) | Two-stage base: P(BLOCK copied from SCHED) classifier plus an L2 regressor on normal flights, mixed by expectation. LOBT ±3606 s window projection. Cross-fitted residual "corrector" ensemble over 2 bases. **NEW, high value** |
| 2 | github.com/yookeew/dumb-puzzle | smart-jigsaw (#42, 254.01) | 10-05 | ≥109 | GPL-3.0 | 2 (to 10-01) | Stack plus context corrector, ADS-B combiner, neighbour-state stage, LOBT window, NM lines. **About 40 commits since our 10-01 note** |
| 3 | github.com/skylinkapi/prc-data-challenge-2026-kind-mango | kind-mango (#33, 246.48) | 09-28 | ≥50 | GPL-3.0 | 33 | linear_tree LGB, LIRF regime head, AOBT_3 since v73. Covered |
| 4 | github.com/saurabhdxt-labs/prc-challenge | merry-quicksand (#38, 249.92) | 09-12 | 44 | GPL-3.0 | 6 | Delta target `BLOCK − AOBT_3`. Covered (public code stops at v12) |
| 5 | github.com/col3name/air-data | kind-earthquake (#51, 268.52) | 09-05 | 13 | none found | 6 | CatBoost with traffic windows. Covered |
| 6 | github.com/javidmardanov/PRC-Data-Challenge-2026 | elegant-alligator (#52, 268.56) | 09-15 | 10 | GPL-3.0 | 7 | TimesFM-3 plus residual models. Covered |
| 7 | github.com/Phoenix-Ops-LTD/prc2026-taxiout | zestful-fountain (#59, 273.56) | 09-13 | 24 | GPL-3.0 | 12 | CatBoost residual on MVT−AOBT_3. Covered |
| 8 | github.com/satam2/knowledgeable-helicopter | knowledgeable-helicopter (#65, 275.64) | 09-12 (side branch to 09-27) | 5 | GPL-3.0 | 6 | Three-way proxy regime split. Covered |
| 9 | github.com/TAUIL-Abd-Elilah/prc-data-challenge-2026 | **merry-mushroom (#66, 276.03)** | 10-03 | ≥50 | GPL-3.0 | 0 | Many fixed-blend CatBoost/LGB "experts" on AOBT_3 residuals, with gated Jan/Jul, Nov/Dec, Apr/Oct and May/Sep audits. **NEW** |
| 10 | github.com/ChandanHegde07/OpenAir | likable-eagle (#68, 276.99) | 09-15 | 28 | GPL-3.0 | 28 | Covered |
| 11 | github.com/kagervig/PRC-Challenge-2026 | unique-umbrella (#98, 291.52) | 10-01 | 23 | GPL-3.0 | 4 | LGB; v33–v39 added gate-hold backlog, overdue runway queue, and removed (airport, hour) target encoding. Skimmed 09-21 only |
| 12 | github.com/fenil264/prc-2026-taxi-out | joyful-lemon (#116, 298.86) | 09-09 | 3 | none found | 6 | Notebooks. Covered |
| 13 | github.com/radekacar/joyous-rainbow | joyous-rainbow (#117, 299.23) | 09-25 | 9 | GPL-3.0 | 3 | linear_tree on raw y. README covered |
| 14 | github.com/sergiuv11/prc-data-challenge-2026 | jubilant-vase (#118, 299.29) | 09-11 | 1 | GPL-3.0 | 17 | Covered |
| 15 | github.com/ahmetabdullahgultekin/prc-taxiout-2026 | unclear (the repo names keen-hamburger and vibrant-lollipop, #129 310.55) | 09-01 | 29 | GPL-3.0 | 11 | EUROCONTROL additional-taxi-time reference plus residual. Covered |
| 16 | github.com/sqcm9896/prc-2026-likable-river | likable-river (#134, 315.81) | 09-09 | 12 | GPL-3.0 | 2 | Covered |
| 17 | github.com/RavindraTarunokusumo/prc-data-challenge-2026 | genuine-cabbage (#158, 364.71) | 10-05 | ≥217 | none found | 3 | Autonomous "Claude researcher plus Advisor" governance experiment, frozen on day 7. Low value |
| 18 | github.com/rocoroza11/nimble-barn-prc-data-challenge-2026 | nimble-barn (#212, 609.62) | 09-17 | 2 | AGPL-3.0 | 0 | Stub `main.py`. NEW, no value |
| 19 | github.com/GLAPM/prc-2026-sincere-donkey | sincere-donkey (#219, 625.43) | 09-06 | 1 | GPL-3.0 | 2 | LGB+XGB+CB. Covered |
| 20 | github.com/elna4os/prc_2026 | incredible-rose (#220, 633.85) | 10-05 | 4 | none found | 0 | Baseline notebook. NEW, no value |
| 21 | github.com/heubert-69/PRCDataChallengeSubmission_QuickPanda | quick-panda (#225, 664.84) | 10-04 | 5 | MIT | 0 | Ridge/EN/GBDT stack with combinatorial purged CV; HMM dropped. NEW, no value |
| 22 | github.com/victoralcadi/prc-data-challenge-2026 | eager-jungle (#227, 689.69) | 09-17 | 12 | GPL-3.0 | 4 | LGB anchored on takeoff. Covered |
| 23 | github.com/AbdullahKhan-x/PRC-2026 | logical-ladder (#121, 303.46) | 09-19 | 1 | GPL-3.0 | 2 | Download scripts only |
| 24 | github.com/MykhailoHalchenko/Asterion-5G | none (not a team entry) | 09-30 | 9 | none found | 0 | 5G/ASTERIX simulator with a TensorFlow taxi-time model on PRC columns. Irrelevant |
| 25 | github.com/eminntrhnn/prc-data-challenge-2026 | unknown | 10-03 | 1 | GPL-3.0 | 0 | README only |
| 26 | github.com/Necklacesaura/PRC-Data-Challenge-2026 | unknown | 09-26 | 11 | GPL-3.0 | 1 | Scaffolding |
| 27 | github.com/Enzoferrariz/PRC-Data-Challenge-2026 | unknown | 09-01 | 1 | none | 1 | Empty |
| 28 | github.com/ruchir321/prc-data-challenge-2026 · github.com/georgejhanlon/prc-taxiout-2026 | unknown | n/a | 0 | none | 1–2 | Empty |
| 29 | github.com/victorvb22/taxi-out-prediction | unknown | n/a | n/a | n/a | 1 | Gone (needs authentication) |
| 30 | github.com/LeeMarshall1113/prc-taxiout-2026 | jolly-lobster (#4, 217.72) per the search snippet | n/a | n/a | n/a | 3 | Private or deleted, cannot be read |
| 31 | github.com/arquicanedo/prc-challenge-2026 | leaderboard viewer | 09-21 | 13 | MIT | 3 | Infrastructure |
| 32 | github.com/euctrl-pru/prc_data_challenge_website_2026 | organiser site | 10-04 | n/a | n/a | 3 | Team pages, no repo links |

## Deep skim: three highest-value repos not yet covered

### A. outgoing-boat (#31, 243.95, 0.6 s behind us)
Solo team, Portuguese-language diary. Stopped work on 10-03 at v37. Our docs only cite its "84.7 %" line. Ideas we have not exploited:
1. **Expectation mixture for SCHED-copy rows.** A classifier gives p = P(|BLOCK − SCHED| ≤ 60 s). The prediction is `p·(MVT − SCHED) + (1 − p)·ŷ_normal`, never argmax. The regressor trains only on non-copy rows, with the target clipped at 7200 s. This is the step that took them from 384 to 339 (v2→v3). It applies to all airports, not only LIRF. The classifier is well calibrated: mean OOF p is 0.0949 against a true rate of 0.0972.
2. **Cross-fitted base for the residual corrector.** The base is refit on 2-month blocks, so the corrector learns the error the base makes on unseen data, not its training error. Together with the LOBT window this took them from 314.8 to 275.9.
3. **Transfer gate.** The RMSE gain decomposes exactly into a sum over flights. They report the share from the body (y ≤ 3600) against the tail. Backtested on v30→v37, the **body share predicts the official delta with ratio 1.03 and MAE 0.15 s** (Pearson 0.985). The tail share is *anti*-correlated (−0.86). A third reading reweights the holdout to 2026 using importance weights from an adversarial classifier (AUC 0.866). This addresses our known problem (e), where CV gains do not transfer.
4. **Feature blocks we may lack:**
   - `ret_*` gate retention: departures past EOBT that have not yet pushed, and EWMA of departures.
   - `pista_*` runway configuration and recent change, wake-weighted queue on the same runway, cadence of the last 20 takeoffs.
   - `sup_*` aircraft on the surface at the *estimated* push-back (MVT − pred).
   - (airport × stand × runway) cell P10, P90 and spread.
   - X-Plane `apt.dat` taxi distance (GPL data).
5. **Training-set hygiene.** `--corretor-sem-regra` keeps rows already served by a fixed rule (NM-missing lines, the Rome rule) out of corrector training.
6. **Negative results worth not repeating:**
   - Seed averaging: 0.2 s, noise.
   - Residual target on MVT−AOBT_3: −0.2 s for them (but with a different base).
   - Multiclass "which timestamp was copied": −4 s.
   - Isotonic calibration of p: no gain.
   - Arrival mirror: |r| ≤ 0.055.
   - Group-mean features (OpenFE): adversarial AUC 0.91–0.95, so drift comes from the operator, not the data.
   - Transductive pseudo-labelling of 2026 rows at weight 0.3: built 10-02, never shown to help.

### B. smart-jigsaw (#42, 254.01): what changed since our 10-01 note (PROGRESS §70–80)
- **Neighbour-state stage.** The residual is shared in time: correlation of 0.30–0.44 with the true residual of other departures within ±15 min, and higher on the same runway. Label-free proxies: neighbours' NM pseudo-residual (T − AOBT_3 − s) and neighbours' ADS-B pseudo-residual. Cross-fit screens show −1.1 to −1.5 % residual RMS.
- **Runway-state corrector block (v36), "idea from outgoing-boat".** It needs a seam guard: gaps over 6 h are set to null because the ranking frame jumps from 31 Jan to 1 Jul. Measured −0.95 on the board.
- **Bug worth checking in our code (v37, −0.50 on the board).** Polars `Categorical.to_physical()` codes follow order of first appearance, so stand-prefix codes were scrambled between the training frame and the ranking frame. The holdout cannot see this because holdout rows sit inside the training frame.
- **Diagnostics:**
  - About 20 teams made a single ~15-point jump between 23 Sep and 2 Oct; the cause is not found.
  - January 2026 is a winter-storm month (376 snow hours against 84 in Jan 2025). MVT−SCHED > 2 h roughly doubles, and EHAM NM-missing goes from 1.6 % to 5.0 %.
  - OPDI `exit-parking_position` covers mainly LSZH and fires about 5 min late.

### C. merry-mushroom (#66, 276.03)
Score is weaker, but it has disciplined gating and two ideas new to us:
- **Neighbour-proxy congestion.** 24 features from other departures' released AOBT_3 values (count, mean, spread by airport or runway, strict past/future 15/60-min windows). In a matched ablation they gave **−1.27 s** (CI 0.93–1.66).
- **Same-runway arrival headways** (8 features). Very deep CatBoost (depth 10, about 10k GPU trees) with a fixed 0.5 blend.
- A Rome/Istanbul "scheduled-time source" classifier with a fixed half-probability correction.
- Extra validation folds (Nov/Dec forward, Apr/Oct, May/Sep) that must all pass.
- Rejected:
  - Movement-ID neighbourhood, as opaque ingestion order; also against our rules.
  - Airport-specific blend weights, which reversed going forward.
  - Historical flight analogue for missing NM rows.
- **Caution:** their "timestamp precision" features (second/minute rounding of released clocks) are close to our banned side channel (second jitter). Do not adopt.

## Notes
- All substantive repos are GPL-3.0 except col3name (no licence file), QuickPanda (MIT) and nimble-barn (AGPL). Use ideas only, no code.
- genuine-cabbage, Asterion-5G, incredible-rose, quick-panda, nimble-barn and eminntrhnn add nothing.
