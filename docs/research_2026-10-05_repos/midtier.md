# Mid-tier public repos: re-survey 2026-10-05 (ideas only, no code copied)

## 0. What is new since the 09-21 notes
Not much. None of the 11 repos has a model commit after 09-17. The one commit after that is arquicanedo's 09-21 leaderboard-viewer change. Every HEAD equals the commit our notes 06–10, 21, 23, 24 and the 09-30 deep dive already read. So the useful work was to re-mine these repos for the six focus topics and check each idea against v37 code (`src/prc`, `scripts/adsb_*`).

| repo | team / LB 10-05 | last commit | licence | notes |
|---|---|---|---|---|
| Phoenix-Ops-LTD/prc2026-taxiout | zestful-fountain #59, 273.56 | 09-13 `4477d9b` | GPLv3 | covered (07) |
| ChandanHegde07/OpenAir | likable-eagle #68, 276.99 | 09-15 `97f603c` | GPLv3 | covered (06) |
| LeeMarshall1113/prc-taxiout-2026 | jolly-lobster #4, 217.72 | — | — | still 404 / not reachable (`add_repo` refused) |
| victoralcadi/… | unidentified | 09-17 `5dccaca` | GPLv3 | scaffold + `docs/papers-and-data.md` (literature, no model) |
| ruchir321/… | — | empty | — | empty repo |
| sqcm9896/prc-2026-likable-river | likable-river #134, 315.81 | 09-09 `866b2f6` | GPLv3 | covered (24); weak |
| arquicanedo/prc-challenge-2026 | quick-boat #7, 222.85 | 09-21 `ba5915f` | MIT | leaderboard viewer only |
| saurabhdxt-labs/prc-challenge | merry-quicksand #38, 249.92 | 09-12 `fdeb704` | GPLv3 | covered (21, deep dive); public state is v12 = 266.37 |
| sergiuv11/… | jubilant-vase #118, 299.29 | 09-11 `e3b653b` | GPLv3 | covered (09) |
| javidmardanov/… | elegant-alligator #52, 268.56 | 09-15 `845d0d1` | GPLv3 | covered (08) |
| satam2/knowledgeable-helicopter | #65, 275.64 | 09-12 `9e5b9e6` | GPLv3 | covered (23); MVT_ID-order features break our rules |

Aside: `src/enio` (outgoing-boat #31, 243.95, GPLv3, closed 10-03) is not in my list but is the only fresh repo; two of its method notes are cited below.

## 1. Focus topics: what these repos add beyond v37

**Unmatched rows (no NM).**
- OpenAir (L2/L4/L5, `experiments/run_e74_make_v20.py`, `run_e98_make_v23.py`, `run_e110_make_v24.py`):
  - learns G = BLOCK − SCHED on **all ~160k LIRF departures**, matched rows included, then uses T = D − Ĝ;
  - soft-blends that with an unmatched-only G at λ = 0.20–0.25. λ = 1 lost 12 s on the LB;
  - "long-hold pull": when OPDI ground time is ≥ 2 h, T is pulled halfway toward a typical ~1.1 ks taxi.
  - Our model B trains on unmatched rows only (~22k) and has the LIRF sched reparam. No matched-trained G member exists.
- merry-quicksand (`reports/STRATUM_MONSTERS.md`, `plans/PREREG_unm_congestion_*`):
  - the unmatched body is fitted on non-fill rows, winsorised at ±3,000 s for the fit only;
  - an expectation mixture with the fill probability, p·(MVT−SCHED) + (1−p)·body, at every airport;
  - an airport-hour matched-proxy witness: −1,775 board MSE. We already have the equivalent (`nbr_mvt_minus_aobt_mean`).

**LIRF/LTFM/LFPG.**
- merry-quicksand `NM_PARAMETER_DIAGNOSTIC`: corrections along NM table-minute axes at LTFM/LEBL/LEMD/EDDM **do not transfer**. The airport-month residual level flips sign Jan↔Jul (LTFM −41/+27 s), and the in-sample oracle is only ~90–140 MSE. Do not spend time there.
- `CLOCK_ARTIFACTS` M3: LTFM BLOCK sits a median +296 s after AOBT_3, which is a definition offset, not a fill.
- `ADSB_REGIME_CHECK`: LTFM and LFPG hold 2,915 of the 7,054 oracle MSE on "early off-block" rows, and ADS-B coverage there is ≤ 6%.

**Copied/defective block times.**
- Known families (LIRF fills, 24 h date-slip, EHAM/LSZH heliports) are already in our docs.
- New nuance: merry-quicksand's date-slip lane transferred at only **0.15×** its fold projection (`MSE_LEDGER`, v7). Extreme-row lanes deserve a heavy haircut.

**ADS-B robust to the 2025→2026 receiver change.**
- merry-quicksand's ADN stacker transferred at **1.36×** projection (v11, −7,722 MSE). It shipped per airport behind a density-ratio check, "RWC" (ledger 2026-09-11 16:26):
  - a 2025-vs-2026 classifier on the ADS-B features within each coverage stratum;
  - the 2025 OOF gain is reweighted to the 2026 mix, with an ESS bar;
  - the KS-hold rule is "hold an airport only if KS(2026 vs 2025) > KS(Jan vs Jul 2025)".
- Its features are expressed **relative to AOBT_3 and MVT** (`sensor_rel_aobt3`, `first_ground_rel_aobt3`, `dwell_end_rel_mvt`), not as fix counts. They also carry a coverage code: 2 = joined mid-taxi, 3 = seen at the stand.
- Post-ADN, **70% of the error ADS-B can still reach sits on code-2 rows**. These rows are first seen a median of +162 s after AOBT_3 (`POST_ADN_ERROR_MAP §6`).
- `ADSB_GATE.md`: stand centroids are learned from ADS-B positions at the recorded off-block, leave-one-flight-out, with no OSM.
  - The estimator "last fix within R of own stand" gives EHAM RMSE 127 against the model's 173, on 66% of rows.
  - EDDM has 73% coverage.

**OPDI.**
- merry-quicksand `OPDI_FEASIBILITY` (6–13 Jan 2025): same-flight OPDI events are usable only at LSZH (taxiway entry 93%, MAD 30 s) and partly at LEBL.
- EDDF has 0% stand-exit and ≤1% taxiway entry. **This contradicts our gap README's "EDDF 47% upper bound"**, so verify it before investing.
- LIRF/LTFM: 0%.

**Validation that tracks the LB.**
- enio `docs/research/2026-10-03-transferencia.md`:
  - the RMSE gain decomposes exactly per row: ΔRMSE = Σwᵢdᵢ / (Σw·(R_base+R_new));
  - so it splits into body (y ≤ 1 h), tail and top-20 rows;
  - its v37 simulated −0.77 s, of which 20 flights were 129% of the gain; the LB gave −0.02 s;
  - adds IWCV weights from a 2025/2026 classifier (AUC 0.866).
- enio `2026-10-03-fe-auto.md`: **group-by aggregates computed inside the frame have adversarial AUC 0.91–0.95, while their source columns sit at 0.50–0.55.** That is a plausible mechanism for our L5 (CV −0.32, LB +0.12).
- merry-quicksand prices each fold gain to 2026 by bin: the per-bin 2025 gain × the 2026 bin counts.

**Post-processing.** Nothing new beyond what we already have (LOBT projection, zero floor, overlays).

## 2. Ranked top-8 transferable ideas (not yet in v37)

| # | idea | impact | effort | risk | evidence |
|---|---|---|---|---|---|
| 1 | **ADS-B censoring projection.** Treat the ADS-B observations as an interval, like the LOBT window: (a) seen moving or away from the stand at t_first, so y ≥ MVT − t_first − s_apt; (b) seen stationary at the stand until t_last, so y ≤ MVT − t_last + s_apt. Set the slack s_apt from 2025 at about a 99.9% non-violation rate per airport and coverage code, then project the final blend. | Med (code-2 rows hold 70% of reachable ADS-B error; a bound costs nothing where it is not binding) | 3–4 h | Low–med: needs a per-airport clock offset (LTFM +296 s; ADS-B time vs airport clock) | MQ POST_ADN §6; our LOBT-window precedent; our gap README (18–27% of rows already moving when first seen) |
| 2 | **Adoption gate on 2026-reweighted, body-only gain.** Report three things for every candidate: the body / tail / top-20 split of ΔRMSE; IWCV-weighted gain (2025 vs 2026 classifier on the features the candidate adds); and, for ADS-B lanes, an RWC-style per-airport × coverage-code reweighting with an ESS floor and the KS-hold rule. Reject when the gain is mostly tail or top-20, or when the reweighted interval crosses 0. | Indirect (stops L5-type misses; may ship ADS-B airports we currently hold out) | 3 h | Low | enio transferencia (would have rejected v37); MQ v11 1.36×, v7 0.15× |
| 3 | **Adversarial screen on frame-level aggregates.** Before adopting any group-by or rolling feature (runway sequence, NM forward windows, airport-hour medians), compute its own 2025-vs-2026 AUC and compare it with the AUC of its source columns. Drop it, or rescale it to a within-day rank or percentile, if it is much higher. | Small–med (CV→LB fidelity) | 1–2 h | Low | enio fe-auto §4 (0.91–0.95 against 0.50–0.55); our `adversarial.py` only does the global check |
| 4 | **Data-learned stand anchors for ADS-B.** Build per-stand centroids from 2025 fixes at the recorded off-block (leave-one-flight-out). For stands new in 2026 (EDDF T3), build label-free centroids from 2026 long stationary runs of departures assigned that stand. Feed "last fix within R of own stand" and the distance from the first fix to the stand into the learned residual. | Med (LFPG stand detection is 3% despite 58% ground fixes; EDDF T3) | 5–6 h | Med (receiver density) | MQ ADSB_GATE (EHAM 127 vs 173) |
| 5 | **Matched-trained G member for LIRF unmatched.** Train G = BLOCK − SCHED on all LIRF departures, then blend T = D − Ĝ into model B at λ ≈ 0.2–0.25. The LIRF-unmatched OOF metric decides. | Med on ~380 rows (very heavy in MSE) | 2–3 h | Low–med (matched G is conditioned on \|BLOCK−LOBT\| ≤ 3606) | OpenAir L2/L4 (λ = 1 lost 12 s on the LB) |
| 6 | **Long-hold check on LIRF unmatched.** Where opdi_ground_s ≥ 7200, compare model B's prediction with the empirical median T (~1.1 ks in 2025). If model B tracks D instead, add a capped pull toward it. | Small (≈150 rows) | 1 h | Med (fails in December; Jan + Jul held for OpenAir) | OpenAir L5 (+1.37 s) |
| 7 | **Unmatched body fit on winsorised non-fill rows** (±3 ks for the fit only), with an explicit fill-probability expectation mixture extended beyond LIRF to the airports where fills exist. | Small–med | 3 h | Med (MQ: stratum 1,648 vs 1,794 plain; ours is already 1,632) | MQ S1/E3C; enio expectation mixture |
| 8 | **Re-verify OPDI same-flight coverage before building on it.** Measure EDDF/LSZH exit-parking and taxiway-entry coverage on Jan + Jul 2025 and 2026. Use LSZH only if it holds. The fairness question (a flight's own surveillance) was raised by MQ and kind-mango; the organiser has confirmed the licence, not this use. | Small (LSZH ≈ −3 RMSE at best, per MQ) | 1 h | Low | MQ OPDI_FEASIBILITY vs our gap README |

**Do not do:**
- NM table-minute or "planned AOBT_3" corrections at LTFM/LEBL/LEMD/EDDM (MQ NMD shows they do not transfer);
- MVT_ID or FLIGHT_ID ordering features (satam2, MQ; a rule violation);
- re-shaping fills by date-slip (0.15× transfer);
- METAR stacking for the tail (OpenAir).
