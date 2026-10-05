# Known competitor repos: what changed since late September (read 2026-10-05)

Clones: `scratchpad/repos/src/<owner>_<repo>`. Read only. No code copied. [S] = read in the repo, [X] = my inference.
Our baseline: v37, 242.59, rank 27.

## 1. Status per repo

| repo (team) | LB 10-05 | last commit | new since our notes | licence |
|---|---|---|---|---|
| yookeew/dumb-puzzle (smart-jigsaw) | **#42, 254.01** (v37) | 10-05 | **a lot**: about 70 commits, 270.2 to 254.0 | GPLv3 (added 10-02) |
| skylinkapi/…-kind-mango | **#33, 246.48** (v105) | 09-30 (branch) | plan only. Public best is still v89, 264.34, so **18 s is private** | GPLv3 |
| saurabhdxt-labs/prc-challenge (merry-quicksand) | #38, 249.92 (v55) | 09-12 | **none**. Public code is still v12, 266.37 | GPLv3 |
| ahmetabdullahgultekin/prc-taxiout-2026 (vibrant-lollipop) | #129, 310.55 | 09-09 | none | GPLv3 |
| RavindraTarunokusumo/prc-data-challenge-2026 (genuine-cabbage) | #158, 364.71 (1 upload) | 10-05 | days 3–8, PRs #3–#10: governance logs, LGB+CatBoost; nothing usable | GPLv3 (day-7 branch only, 10-05) |
| **EnioAguiar/prc-taxiout-2026 (outgoing-boat)**, not on the brief's list | **#31, 243.95** (v37) | 10-03 (closed) | **we have never studied it**. smart-jigsaw lifted four levers from it | GPLv3 |

## 2. smart-jigsaw: new commits (09-29 to 10-05) and the LB effect of each [S] (`PROGRESS.md` §52–§80)

| ver | change | LB Δ |
|---|---|---|
| v23 | ADS-B detector v3: **stands inferred from ADS-B dwell clusters** (50 stands missing from Gateway; median 7 m off) + **gated fallback match** for departures whose track has no take-off roll (aircraft sitting at its own stand in [T−90 min, T−2 min], later moving; kept only if \|est − base\| ≤ 300 s) | −0.70 |
| v24 | **Learned ADS-B combiner**: one LGB on (y − stack), from every detector field. Replaces the linear stages | −3.84 |
| v25 | Combiner **applied only to rows that have ADS-B information**; other rows keep the earlier value | −1.27 |
| v26 | Combiner retrained with **Sep–Dec 2025 ADS-B** (44 days, honest OOF base on those months) | −1.69 |
| v27 | Combiner retrained on **all of 2025** (115 days, 660k rows) | −0.57 |
| v28 | LIRF-only ADS-B combiner | +0.23 (rejected) |
| v29 | Neighbour-state corrector, unconstrained | +0.20 (rejected: level drift) |
| v30 | **Within-day neighbour corrector**: the correction is demeaned per airport-day, applied only to rows without ADS-B | −0.73 |
| v31 | **13 model-free neighbour features in the base LGB** (file `src/features/neighbour_feats.py`; listed below the table) | −1.51 |
| v32 | **LOBT window**: \|BLOCK − LOBT\| ≤ 3606 s held on 100 % of 2025 rows; prediction projected into it (idea from outgoing-boat) | −1.84 |
| v33 | **NM-unmatched long-delay lines** y = a + b·(MVT − SCHED) for MVT − SCHED > 6 h, per airport, plus a Rome line (from outgoing-boat) | −0.69 |
| v34 | **OOF context corrector**: LGB on (y − stack), trained on 10 OOF months (detailed below the table) | −0.58 |
| v35 | Fix for drifting cumulative-count features (2025-vs-2026 SMD 0.60). Corrector becomes the mean of a global and a per-airport LGB | −1.32 |
| v36 | Corrector **runway-state block** (detailed below the table) | −0.95 |
| v37 | **Categorical-code bug fix**: Polars `Categorical.to_physical()` codes differed between the train and ranking frames | −0.50 |

**v31 neighbour features.** For other departures within ±15/30 min (by airport and by runway):
- the mean NM taxi (T − AOBT_3);
- the mean delay (T − SOBT);
- each of these minus its own airport-day mean;
- the arrivals' taxi-in excess over a 2025 per-(airport, stand, runway) median.

**v34 corrector inputs:**
- stand prefix;
- leave-one-month-out cell statistics;
- NM consistency;
- distance to the edges of the LOBT window;
- "retention" at AOBT_3: flights past EOBT that have not pushed, active deps on the same runway, mean AOBT_3 − SCHED over the previous 15 min;
- aircraft on the surface at the *estimated* push time, MVT − pred.

**v36 runway-state block:**
- cadence of the last 20 take-offs;
- gaps to the previous and next take-off on the same runway;
- the runway's share of the last hour's take-offs;
- landings on the same runway;
- change of the dominant runway;
- heavies queued on the runway;
- a guard at the seam between 31 Jan and 1 Jul.

**Closed by smart-jigsaw** (skip): far-sighting, runway-end and inbound-hex matching, pseudo-labels, history encodings, LIRF hedge rules (four LB misses), OPDI exit-parking (LSZH only, about 5 min late).

**Rules caution.** smart-jigsaw twice infers an "optimal scale" from two LB scores (§66, §71). That is LB probing; do not imitate.

## 3. outgoing-boat (EnioAguiar), new to us [S] (`README.md`)

- **Base model.** A copy classifier p = P(\|BLOCK − SCHED\| ≤ 60 s), then an L2 regressor trained on non-copy rows only (target clipped at 2 h). They are mixed **by expectation**, p·(MVT − SCHED) + (1 − p)·ŷ. The LOBT window also sets p to 0 when SCHED falls outside it.
- **Cross-fitted corrector.** Base OOF predictions come from **2-month blocks over the whole year**. Four corrector members, each the mean of a global LGB, a per-airport LGB and CatBoost, combined with equal weights over two bases. Correctors skip the rows served by a rule (`--corretor-sem-regra`). v6 to v9 went 314.8 → 275.9, the corrector together with the LOBT window.
- **Transfer gate** (`src/transferencia.py`). The RMSE gain is decomposed exactly into a per-flight sum. Their claims:
  - the **body share (y ≤ 3600 s)** predicted the LB delta over 6 consecutive uploads with ratio 1.03, MAE 0.15 s, r = 0.985;
  - the tail share is *anti*-correlated with the LB delta (−0.86);
  - the gate adds a cap on the share held by the top 20 flights, and importance weights from a 2025-vs-2026 adversarial model (AUC 0.866).
- **`fe_auto`.** 755 generated candidates; one survived: the **rank of MVT − AOBT_3 within day × airport × hour**.
- **Risk.** They use the EUROCONTROL daily `all_pre_departure_delays` series. On 10-05 our team found it is an aggregated label (corr 0.998). Do not follow them there.

## 4. kind-mango: no new results [S] (`docs/WINNING_PLAN.md` @ 4091dc8)

The plan for 09-30 to 10-11:
- **OPDI own-flight pushback.** Link on `CALLSIGN_flt`, take the *latest* pushback in [MVT − 7200 s, MVT], one row per OPDI flight, and do not link on the first runway entry. Gate: LSZH coverage ≥ 40 % and half the error of the NM clock.
- A one-day ADSB.lol probe.
- They call the model side saturated.

They have since gained 18 s privately, plausibly from an ADS-B or OPDI clock [X].

## 5. What we already have, checked against `src/prc` and `docs`

| lever | ours? |
|---|---|
| LOBT window 3606 s | **yes** (`train.py:296-308`, `lobt_bound: true`) |
| anchored and plain members, CatBoost, per-airport member, MLP | yes |
| backward and forward NM neighbour means, runway sequence and headway features | yes (`nbr_*`, `seq_*`, `rwy_headway*`) |
| learned ADS-B residual applied only on rows with ADS-B | yes, but trained on **62 Jan/Jul 2025 days** (`adsb-learned-residual.md`) |
| deterministic categorical codes | **yes**: `train.py:1203` stacks train and target before an Enum; the ADS-B scripts use fixed `CATS`. No smart-jigsaw-type bug |
| OOF base predictions on 10–12 months (block cross-fit) | **no**: our folds are jan, jul, jul_fwd, forward |
| context or retention corrector, surface at estimated push | no |
| within-day-demeaned neighbour anomalies; neighbours' ADS-B pseudo-residual | no |
| inferred stands, gated fallback ADS-B match | no (gap note: pushback detected on only about 21 % of rows) |
| long-delay unmatched lines outside LIRF | unclear. Model B is a learner, not a line |
| transfer (body-share) gate | no (we use K=200 trimming) |

## 6. Top-8 transferable ideas, ranked

| # | idea | impact (evidence) | effort | risk |
|---|---|---|---|---|
| 1 | **Block cross-fit OOF of the base over all 2025 months** (e.g. six 2-month blocks). Then train the ADS-B residual on the **full year of 2025 ADS-B**, not only 62 days | smart-jigsaw −1.69 then −0.57 LB, 1.2–1.4× holdout; outgoing-boat's whole corrector stack depends on this OOF | high: 6 base refits + about 50–120 ADS-B days of ingest | low; physical signal, transferred about 1:1 |
| 2 | **OOF context corrector** on (y − blend), using the OOF from #1: retention at AOBT_3, surface count at MVT − pred, LOMO cell stats, distance to LOBT edges, NM consistency; mean of global + per-airport LGB | −0.58, −1.32 (after the drift fix), −0.50 LB (smart-jigsaw); 247.8 → 243.9 range (outgoing-boat) | med–high (needs #1) | med: **check each column's 2025-vs-2026 SMD**, since cumulative counts leaked a time index |
| 3 | **Runway-state / retention block** (as a base feature or in the corrector): last-20 cadence, runway share, dominant-runway change, heavies queued, deps past EOBT not pushed | −0.95 LB (smart-jigsaw v36); −0.71 (outgoing-boat v34) | low–med; overlaps our `seq_*` | low |
| 4 | **Within-day neighbour anomalies**: ±15 min airport mean of T − AOBT_3 and T − SOBT *minus the airport-day mean*; arrival taxi-in excess over a per-(stand, runway) median | −1.51 LB as base features (v31) + −0.73 as a demeaned corrector (v30); undemeaned version **failed** (+0.20) | low–med | med: demeaning is what makes it transfer (2025→26 level drift) |
| 5 | **ADS-B coverage: inferred stands + gated fallback match** for tracks with no take-off roll | −0.70 LB (v23), with LSZH −6.4 and EGLL −1.3 on holdout | med | low–med; the ±300 s gate carries it |
| 6 | **Body-share transfer gate** for adoption: score the y ≤ 3600 s part of the paired gain, cap the top-20 share, add an IW-reweighted check | predicted 6 LB deltas, MAE 0.15 s (outgoing-boat). Targets our CV→LB problem (e) | low | low (decision tool) |
| 7 | **Neighbours' ADS-B pseudo-residual** (observed neighbours' ADS-B taxi − lag − base, ±15–60 min) as an input for rows **without** their own pushback | label-free corr 0.13–0.27 with the residual (smart-jigsaw §70); part of v29–v31 | med | med; only on ADS-B airports |
| 8 | **Long-delay lines for NM-unmatched rows outside LIRF** (MVT − SCHED > 6 h, per airport) + rank of MVT − AOBT_3 within day × airport × hour | −0.69 LB on 96 rows (v33); the rank is outgoing-boat's only surviving auto-feature | low | high for the lines (tail; only 7 % of holdout gain transferred); low for the rank |

**Evidence.** All LB deltas are the teams' own uploads, as their repos report them [S]. Their bases differ from ours, so treat the sizes as indicative.
