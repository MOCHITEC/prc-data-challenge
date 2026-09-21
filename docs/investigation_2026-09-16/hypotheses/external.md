# Hypothesis generator: external / top-team practice (subagent output, verbatim, 2026-09-16)

Legend: [S] = backed by a source the agent read; [X] = the agent's extrapolation.

## A1. Official 2026 rules [S]
- Site: https://prc-data-challenge-2026.netlify.app/ (data.html, ranking.html, eligibility.html)
- Ranking = best RMSE across submissions; 5/day; 1 GB; every MVT_ID_mvt must match. "We will monitor submissions for attempts to learn from or exploit the ranking process. We consider such practices unfair and inconsistent with the purposes and goals of the Data Challenge." (ranking.html)
- External data allowed if openly accessible, documented, open-source licensed. Prizes need all code on GitHub under GPLv3. Team size not stated. (eligibility.html)
- 2026 site has NO final-phase text as of 2026-09-16 (raw ranking.qmd and commit log of github.com/euctrl-pru/prc_data_challenge_website_2026 checked). 2025 had Phase 2 with fuel_final_submission.parquet; 2024 reserved 52,190 flights.
- data.html: MVT_TIME_UTC_mvt = "(best available) movement time"; movements LEFT-JOINED to NM flight table; "Real world data is messy… We haven't tried to reconcile them". 10 airports in local ranking set (no LTAI).
- Discord: https://discord.gg/RPh89jpVVz

## A2. Live leaderboard 2026-09-16 [S] (https://prc-leaderboard.fly.dev/data.json)
154 teams. vigorous-whistle 243.58 (13 subs), enthusiastic-daisy 244.83 (49), youthful-giraffe 245.02 (21, last 09-10), jovial-uniform 245.29, gentle-igloo 252.43, gentle-lemon 259.00, merry-quicksand 260.37, quick-boat 260.81, zesty-puzzle 263.63, strong-crane 264.20. resilient-hedgehog 305.03.
youthful-giraffe: v3–v12 264.4→262.7, then v13 = 248.5 on 09-08 (one-step −14 s), then 245.02 by 09-10. Solo.
  RETRACTED 2026-09-21 by docs/competitor_research_2026-09-21/results/02-motoki-method.md: this line previously named the
  team as Matt Motoki. That claim carried no URL and no source was ever found for it. The official team page
  https://prc-data-challenge-2026.netlify.app/teams/youthful-giraffe.html gives Indonesia, Independent, 1 member,
  registered 2026-09-01. Motoki is US-based. The submission curve above is also unverified: it appears in no snapshot
  we hold and the live feed no longer exposes per-submission history through this endpoint.

## A3. Public 2026 repos [S]
- OpenAir / ChandanHegde07, LB 276.99 — https://github.com/ChandanHegde07/OpenAir
  Stage1 per-airport OLS on MVT−AOBT and AOBT−EOBT; Stage2 LightGBM on residual with "disruption state" (mean peer lateness 30-min, self-excluded); Stage3 NNLS blend CatBoost 45.6% + airport LGB experts 49.1% + XGB 5.3%; Stage4 LIRF-unmatched gate-delay decomposition T = D − G, D = MVT − SCHED, G = gate hold predicted by CatBoost on all LIRF rows with taxi-in / stand-turnaround features; Stage5 OPDI ground time ≥7200 s → 50% toward median (~146 rows).
  Matched RMSE plateau ≈236 while LB 277. Progression 316.97 → 288.90 (LIRF gate-delay) → 281.72 (taxi-in features) → 278.36 (OPDI) → 276.99.
  Falsified by them: rolling queue features did not shrink tail; previous-departure block intervals corr≈0.008; unmatched LFPG "nearly random".
- zestful-fountain / Phoenix-Ops-LTD, LB 273.56 — https://github.com/Phoenix-Ops-LTD/prc2026-taxiout (METHOD.md)
  CatBoost 183 features + LightGBM + specialists. v7 arrival features (26 taxi-in/same-stand features from ARR rows) −2.6; v8 deeper arrival residual −7.1; v9 NM-neighbour peer MVT−AOBT_3 proxies (30 features) −0.9; v10 CatBoost −2.6; v11 clock-overlay rule (AOBT_3−LOBT > 7200 and both proxies in [−7200, 172800] → predict max(0, MVT−LOBT); 21–31 rows) −0.2; v12 "following-context" NM features (windows strictly AFTER query time, 15/60 min) −1.4. LIRF-unmatched specialist CatBoost on 1,488 rows, 5-seed avg.
- javidmardanov, LB 268.56 — https://github.com/javidmardanov/PRC-Data-Challenge-2026 : TimesFM airport series + CatBoost residual + missing-timestamp mixture of experts + winter-only Rome flight-identity correction.
- sergiuv11 / jubilant-vase, LB 286.66 — https://github.com/sergiuv11/prc-data-challenge-2026 : single global LightGBM (127 leaves, min_leaf 200, 436 trees) + LIRF overlay at 14,400 s + airport specialists blended 50/50.
- ahmetabdullahgultekin — https://github.com/ahmetabdullahgultekin/prc-taxiout-2026 : P10 unimpeded reference per (airport, stand, runway) per ATXOT; IEM ASOS/METAR; OurAirports. Probe: MVT_TIME zero-second share ~8.3% at EDDM/EGLL/LFPG/LIRF/LTFM vs ~1.7% elsewhere (locally confirmed 0.081–0.085 vs 0.015–0.018). Jan/Jul 2025 mean 995/1026 s, SD 605/745 vs mid-year; LIRF SD 1,332 s.

## A4. Past editions [S]
2024: 1st likable-jelly LightGBM + physics + METAR + normalised target; 3rd tiny-rainbow CatBoost+LGBM+XGB+MLP+SAINT stacked. 2025: 1st resourceful-quiver 200.83 LightGBM + domain knowledge; 2nd jubilant-vase CatBoost 64%/LGBM 20%/XGB 16%, GroupKFold by flight.

## A5. Literature [S]
ATXOT: SUPERSEDED. The description here (median per runway x stand group, >=10 flights, sub-saturation
  congestion, taxi-out >=300 min excluded) is Edition 00-04. Edition 01.00 (2023-03),
  https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf, replaced it: 10th PERCENTILE, not
  median; stand individually x runway, not stand group; at least 10 flights AT OR BELOW that P10; a rolling
  12 months; exclusions are taxi-out > 120 min, helicopters, de-icing after AOBT, and missing runway/stand/
  off-block. The saturation-level mechanism is gone. Corrected 2026-09-21 by
  docs/competitor_research_2026-09-21/results/19-atm-literature-sota.md. NASA Charlotte: RMSE 5–6.5 min. CDG regression tree RMSE 1.6 min; top features unimpeded time, congestion, departures last 20 min. Shenzhen stacking: RMSE 140 s after discarding taxi >3600 s. NASA pushback winners: all per-airport models.

## Part B — 12 candidates (unranked)
H1  unmatched-LIRF error dominant (OpenAir matched ≈236 at LB 277 vs ours 217.7 at LB 305). Verify: SSE share of unmatched rows by airport on OOF. No training.
H2  gate-hold decomposition for unmatched rows T = (MVT − SCHED) − G. Verify: corr(TAXITIME, MVT−SCHED) on LIRF unmatched vs LFPG unmatched. Then training run for G.
H3  arrival/turnaround features from ARR rows (taxi-in, in-block, previous arrival same stand). Verify: join DEP→latest ARR same airport+stand; coverage and corr with y by matched/unmatched. Training run to score.
H4  peer "following-context" NM clock features (mean peer MVT−AOBT_3, AOBT_3−EOBT_1, ±30 min, LOO, incl. windows after query time). Verify: corr with model-A residual. Training run.
H5  rule overlays for extreme rows (AOBT_3−LOBT > 7200 → max(0, MVT−LOBT); ground ≥7200 → shrink). Verify without training on training rows.
H6  tail post-processing of model B (cap / shrink toward airport-hour median). Verify on existing OOF: RMSE vs cap quantile; ranking predictions >4h vs training rate.
H7  model B on all rows (shared_b). Pre-check: per-airport y distribution unmatched vs matched. One training run.
H8  airport-specialist models + CatBoost/LGBM NNLS blend. Pre-check: OOF residual by airport × month. Training runs.
H9  explicit clock-difference features AOBT_3−EOBT_1, LOBT−EOBT_1, IOBT−EOBT_1; P10 reference per (airport, stand, runway). Verify: grep features.py; corr with residual. Training run.
H10 timestamp-precision clusters (MVT_TIME second==0 share differs by airport). Verify: OOF residual SD by (airport, second==0).
H11 Jan/Jul seasonality shift (mean/SD elevated; de-icing). Verify: per-month OOF RMSE; residual by airport Jan vs rest; re-measure weather with fixed join.
H12 procedural: public logs show 2–28 s gains from structural changes, ≤1.4 s from single features; screening resolution 0.25 s cannot see structural wins.
