# smart-jigsaw (yookeew/dumb-puzzle): full experiment history

Read 2026-10-05, origin/main @45572fa (10-05 10:50); branch adsbround3 read too. GPLv3. Ideas only, no code copied.

Notation: H = full-holdout Δ (Jan+Jul 2025, s), T = trimmed (labels ≤ 5 h), B = body (NM-matched, y ≤ 1 h). K = kept, R = rejected. ⚠ = LB-probing element, excluded from our use.

Their target is taxi itself, or d = BLOCK − SOBT ("flip"). Their holdout RMSE runs about 1.25× their LB, so the same MSE change looks about 1.25× bigger on the LB.

## 1. Every logged experiment

| § | change | holdout Δ | LB Δ (ver) | fate |
|---|---|---|---|---|
| 1 | config-change reset across the Jan/Jul seam; isolated Jan+Jul holdout | +7.8 | 508 (v5) | K, fix |
| 3 | Huber α=800 | −6.1 (eval later shown buggy) | −7 (v6) | K |
| 6 | stop excluding taxi > 4 h from the eval / ceiling 140k on all rows | 303→523 / +38 | – | K / R |
| 7–8 | echo-lane blend, then a P(echo) classifier blended into every row | −38, −87 | −41 (v7), −84 (v9) | K |
| 9–11 | direct target (−13, LIRF +55, R); "mixed" target: flip for LIRF, direct elsewhere | −27 | −33 (v11) | K |
| 12 | per-row ceiling 140k at LIRF only | −32.5 | −41 (v12) | K |
| 13–23 | rejected, all R: hour echo encoding +6.8; schedule pressure +3.1; echo routing (isotonic, closed-form) +9.7/−0.04; log target +1.3; per-airport heads (Jan↔Jul r −0.3…−0.9); arrival-disruption +0.5; ATFM daily −0.4 ±8 (ceiling 2.07); ADES +15 direct / −0.07 mixed; L2 instead of Huber +0.79; 4 seeds −0.32; OSM routed distance (ceiling 4.8); AOBT_3-anchored queue +1.8 | | – | R |
| 24–28 | METAR, 17 columns (+4.9 under direct, −1.2/−1.7 under mixed); `minute_of_day` Int8 fix (−0.3); cyclical day-of-year +0.37 (R) | | in v15 | K |
| 29–36 | rejected, all R: naive ADS-B dwell detector (RMSE 2,180); LIRF-only model (LIRF +17); echo_prob suppression on NM-null rows; LABEL_HI 200k +6.0; tail router +30.9 (oracle −12.4) | | – | R |
| 37–38 | stand-aware appear/dwell detector, Gateway stands (RMSE 198 vs AOBT_3 654): K. Straight-line stand→runway geometry +0.59: R | | – | |
| 40 | LGB+CatBoost NNLS stack (with weather and minute fix) | −4.5 | −6 (v15) | K |
| 39/40 | ADS-B tier blend with per-airport lag | −3.1 | −19 (v16) | K |
| 41 | partial track: first seen < 1 km from the stand | −0.45 | −2 (v17) | K |
| 42 | EGLL lag +59 s from AOBT_3 drift ⚠ | none (board A/B only) | +1 (v18) | R |
| 43 | ADS-B weights modulated by detection quality | −0.47 | −0.75 (v19) | K |
| 44–46 | LIRF "+24 h" rule (11 rows) | −4.0 | **+3.75** (v20) | R |
| 49 | OOF corrector v2: Huber, rows selected on prediction ≤ 7,200, NM-unmatched operator echo rate | −2.76 / T −3.25 | −4.05 (v21) | K |
| 55 | LIRF 1–6 h hedge shift | −0.77 | **+0.66** (v22) | R |
| 56 | ADS-B v3: stands inferred from dwell clusters + gated fallback match | T −0.62 / −0.50 | −0.70 (v23) | K |
| 58 | learned ADS-B combiner (LGB on y − stack) | T −2.63 / −2.08 | −3.84 (v24) | K |
| 58 | combiner applied only to rows with ADS-B ⚠ decided on board | rows without ADS-B +2.0 | −1.27 (v25) | K |
| 42–80 | rejected, all R: LIRF ADS-B echo mixture −0.04; CatBoost 20k −0.11; corrector v1 (L2, label-selected) +6.2; LGB/joint corrector (Jul worse); far sightings, runway-end, inbound-hex, pseudo-label, overnight scan (killed); delay-band cells +0.5; flight-number and history encodings (worse); ADS-B base features (−0.9 via chain); stand-free v6; single-month and 3-model correctors; OPDI (LSZH 92 %, 5 min late) | | – | R |
| 64 | combiner + Sep–Dec 2025 ADS-B (44 days, OOF base) | T −1.22 / −0.98 | −1.69 (v26) | K |
| 65 | combiner + all 2025 (115 days, 660k rows) | T −0.48 / −0.38 | −0.57 (v27) | K |
| 66 | LIRF-only ADS-B combiner ⚠ | Jul T −0.99 | **+0.23** (v28) | R |
| 71 | neighbour-state stage, unconstrained ⚠ | T −3.78 / −2.20 | **+0.20** (v29) | R |
| 72 | neighbour corrector demeaned within the airport-day, rows without ADS-B ⚠ | T −3.37 / −1.93 | −0.73 (v30) | K |
| 74 | 13 model-free neighbour features in the base LGB | T −1.36 / −1.05 | −1.51 (v31) | K |
| 75 | LOBT window ±3,606 s | not gated | −1.84 (v32) | K |
| 76 | NM-unmatched long-delay lines + Rome line | −9.7, T +0.03 | −0.69 (v33) | K |
| 77 | OOF context corrector, 6 blocks | B −2.94 / T −1.25 / −1.08 | −0.58 (v34) | K |
| 78 | fix for cumulative-count drift (SMD 0.60) + mean of global and per-airport LGB | B −1.60 / T −0.99 / −0.74 | −1.32 (v35) | K |
| 79 | runway-state block | B −0.46 / T −0.59 / −0.49 | −0.95 (v36) | K |
| 80 | Polars categorical-code mismatch fix | 0 (invisible on holdout) | −0.50 (v37) | K |

## 2. Holdout delta vs LB delta

There are 24 pairs where both are known.

**Sign agreed in 20 of 24 (83 %).** The four flips:
- v20 (+24 h) and v22 (hedge): LIRF tail rules learned on 2025.
- v28: LIRF ADS-B. The gate used Jul 2025; the deployed rows were Jan 2026.
- v29: neighbour inputs whose level drifts from 2025 to 2026.

**Size gives no protection.** The flips had holdout gains of 0.8–4.0 s. Gains under 0.5 s transferred (v17, v19, v23, v27, v36).

**Ratio of LB Δ to holdout Δ:**
- Median about 1.5× (about 1.2× after the RMSE-scale factor).
- ADS-B stages: 1.4–6×, because 2026 coverage is higher.
- Context and neighbour correctors: 0.2–0.5×.
- NM lines: 0.07×.

**Transferred at 1× or more:**
- physical observation, and more ADS-B training days;
- a second tree family, and the OOF corrector;
- base features learned on all months;
- drift and encoding fixes (v35 1.3×, v37 seen only on the board).

**Transferred weakly or reversed:**
- correctors with drifting inputs;
- tail rules;
- anything LIRF-specific.

Even with their final gate, B→LB still ran 0.2–2×. That gate: B decides, T and full RMSE are guards, Jan and Jul must both improve, and every new column must pass |SMD| ≤ 0.25.

**For our CV→LB problem:**
1. Add a 2025-vs-2026 SMD screen and a check that the train and ranking frames produce identical categorical codes. These catch what folds cannot.
2. Our L5 miss (CV −0.32 → LB +0.12) is in the size band where their changes did transfer. Suspect drift in L5's inputs, not the size of the gain.

## 3. Negative results we can skip

Every R row in §1. The most useful to avoid:
- daily external data (ceiling about 2 s);
- geometry and OSM distance;
- seeds;
- label-cap and tail routers;
- echo reweighting;
- identity and history encodings;
- stacked correctors;
- all LIRF rules;
- ADS-B as base features, far sightings, runway-end and hex matching;
- single-month correctors;
- METAR in the corrector;
- OPDI.

## 4. Ideas with LB evidence that are not in our pipeline

Checked against `src/prc` and the v37 configuration. We already have: anchored and plain members, CatBoost, the LOBT window, `nbr_*` raw means, rich `seq_*` features, `bk_*` backlog, `planned_flight_s`, `diverted`, a learned ADS-B residual restricted to rows with ADS-B, and the robust-loss test (done; harmful for us).

Ranked by their LB Δ:

| # | idea | their LB | how we measure it |
|---|---|---|---|
| 1 | **OOF corrector on base predictions over the 10 non-holdout months.** LGB with Huber; rows selected on prediction ≤ 7,200 s, never on the label; plus the per-(airport, operator) echo rate among NM-unmatched rows | −4.05 | Needs base OOF for Feb–Jun and Aug–Dec (6 two-month blocks). Gate on jan/jul folds: B and T, both months |
| 2 | **Context-corrector blocks** on the same OOF residual, mean of a global and a per-airport LGB. Blocks: retention at AOBT_3 (flights past EOBT_1 not pushed; active departures on the runway; mean AOBT_3 − SCHED over the last 15 min), surface count at T − pred, leave-one-month-out cell stats, pred − (T − X) for each planned time X, ARVT_3 − AOBT_3 vs planned duration, stand prefix | −0.58, −1.32, −0.50 | Same OOF base. Check every column's SMD ≤ 0.25 and build codes from a sorted 2025+2026 vocabulary |
| 3 | **ADS-B combiner trained on full-year 2025 ADS-B** (we use 62 Jan/Jul days) | −1.69, −0.57 | Our `adsb_learn_*` on 44, then 115 more days with OOF base. Gate on ADS-B rows: B and T |
| 4 | **Within-day neighbour anomalies as base features**: ±15/30 min mean NM taxi and delay, each minus its airport-day mean; arrivals' taxi-in minus the 2025 (airport, stand, runway) median (ours is a raw mean) | −1.51 | Add to the LGB members; jan/jul/forward folds; SMD screen |
| 5 | **Runway-state block**: cadence of the last 20 take-offs; the runway's share of the last hour; change of dominant runway; heavies queued on the runway; null across the 31 Jan / 1 Jul seam | −0.95 | Check first for overlap with `seq_*` and `seq_rwy_share_pm30`. Put it in the corrector (#1) or the base |
| 6 | **Neighbour corrector demeaned per airport-day**, applied only to rows without ADS-B | −0.73 | Correction forced to zero mean per (airport, day); on folds |
| 7 | **ADS-B v3**: stands inferred from dwell clusters (median 7 m error) + fallback match (own-stand dwell in [T−90, T−2] min, kept only if \|est − base\| ≤ 300 s) | −0.70 | Count new detections on 2026 days, then the combiner gate. Our gap note says pushback is detected on only ~21 % of rows |
| 8 | Quality-modulated weights; partial-track nudge (first seen < 1 km) | −0.75, −2 | Probably absorbed by our learned residual. Verify that first-sighting distance, ground speed and gap are inputs |
| 9 | NM-unmatched lines y = a + b(MVT − SCHED) for delays > 6 h, outside LIRF | −0.69 (7 % of holdout gain) | Compare with Model B on the unmatched rows. Tail risk is high |

## 5. 2025 → 2026 differences they quantified

**ADS-B observed-pushback share, Jan/Jul 2025 → 2026:**

| airport | 2025 | 2026 |
|---|---|---|
| EGLL | 2.6 | 49 |
| LEMD | 0.3 | 18 |
| LSZH | 20 | 28 |
| EHAM | 66 | 74 |
| LEBL | 30 | 41 |
| EDDM | 48 | 76 |
| EDDF | 12.4 | 4.2 |

- EDDF falls because of reception: tracks are first seen 1–2 km out.
- LIRF is 14 % (Jan) and 0 % (Jul) in 2026. LFPG and LTFM are 0.
- Partial-track-eligible rows: 13.7 % in ranking vs 11.1 % in the holdout.
- ADS-B − AOBT_3 is stable except at EGLL (−39 to −80 s); the LB said leave it.
- The NM AOBT_3 vs ADS-B gap shrank from 86 to 19 s at LSZH in July.

**Stands:**
- EDDF July 2026 resolves 83.5 % of stands vs 93 % before. The gap is new Terminal 3 stands J1–J10, H5 and H9 (~1,900 rows).
- At LSZH, the Gateway coordinates for stand families B/D/G/C/T look displaced (~5,100 rows).
- Unseen categories: < 0.1 %.

**NM and weather:**
- NM null rate: 1.08 % (2025) vs 1.53 % (2026).
- Jan 2026 was a storm month: 376 snow hours vs 84.
  - MVT − SCHED > 2 h roughly doubled at EHAM, LSZH, LTFM, EDDF and EDDM.
  - EHAM's NM-missing rate went from 1.6 % to 5.0 %.
- Weather is the only base family that shifts by more than 0.3 sd.

**Neighbour levels drift:** LTFM pseudo-residual +262 → +164; LFPG −87 → +44; January delays up ~500 s at LTFM and EDDF.

**LIRF:** arrival echo is unchanged (8.6/8.7 % in Jan, 10.6/8.9 % in Jul).

**Excluded as LB probing:**
- §66, §71 and §72: scale factors inferred from two LB scores;
- v18: tested on the board only;
- v25: decided on the board only, though a holdout signal supports the idea.
