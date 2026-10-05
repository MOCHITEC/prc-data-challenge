# Referee report: methods and numerical correctness (F1–F4)

Recomputed independently from `data/raw/training_2025-*.parquet` (my code: `scratchpad/review/r1–r7.py`). 2,085,026 departures with a local year of 2025. No duplicate MVT_ID; 6 duplicate FLIGHT_IDs (negligible). TAXITIME_SEC equals MVT−BLOCK on every row. The time-zone handling (local month taken from the runway time, as PRU does) is correct, and so are the 120-min cap (≤7200 s kept) and the P10 with the "≥10 flights ≤ P10" rule, which match the PRU document.

## F1 – replication of the PRU ATXOT. Verdict: holds with caveats
- Recomputed: pooled r = **0.9926**, MAD 0.099 min (n = 120 airport-months) with the METAR de-icing proxy. Without the proxy: **r = 0.937**, MAD 0.21. Per-airport monthly r with the proxy runs from 0.80 (EHAM) to 1.00, and EHAM has a 1.04-min miss in one month. Without the proxy, EDDM is −0.04 and LSZH −0.45.
- The pooled r is mostly between-airport variance. After demeaning by airport the r is 0.97 (proxy) and 0.75 (no proxy). Report this figure, not 0.993.
- Annual ranks: EDDF/LEMD swap at 6/7 (confirmed). Monthly airport ranks match PRU in only **69%** of the 120 cells. LIRF is monthly #1 in 7 months in our replication and 6 months in PRU's.
- Deviations from PRU:
  - The reference is FY-2025 for every month. PRU uses a rolling Feb-2024..month-i window, so the early months here use future data. The "ROLL" variant is not equivalent because it has no 2024 data.
  - De-icing is a weather proxy, not the APDF de-icing flag. It over-excludes: at EDDM 87% of flights are kept versus PRU's 94.6% valid (n_ref ratio 0.88); at LSZH 0.93.
  - The F1 headline depends on this proxy. If its thresholds were chosen while looking at the fit, that is a forking path.
- Must do: report within-airport r and per-airport MAD/max error. State how the proxy was fixed (a priori or tuned) and give its sensitivity. Calibrate the proxy against the monthly VALID_FL counts.

## F2 – schedule fill at LIRF and the KPI impact. Verdict: core result holds; numbers mislabelled
- Recomputed: BLOCK=SCHED excess = **15.64%** at LIRF (matches), LEBL 1.6% for the year, EGLL 0.39%, LEMD 0.66%.
- **Labelling error.** The "FILL 9.4%" is `F_PLANFILL`: BLOCK equals any of SCHED, EOBT_1, LOBT or IOBT, and NM is >5 min later or null. The schedule-only fill (`F_SCHEDFILL`) is **7.83%** (7.42% with non-null AOBT_3). The 32.1 → 15.1 min figures and the repair also use PLANFILL. The README calls all of this "予定時刻で穴埋め" (filled with the scheduled time), which is wrong.
- **KPI.** Recomputed 6.80 → **5.50** (repair with PLANFILL and NM time). Schedule-only repair gives 5.65.
- **Is it fill or genuine on-time pushes?** Mostly fill.
  - The 0-minute bin holds 18.1% of flights, against 3.67% at −1 min and 1.25% at +1 min. The depleted +1 bin suggests that near-on-time pushes are also overwritten.
  - In the 0 bin, 41.8% of flights have NM >5 min later. In the neighbouring bins it is ~25%.
  - Of the ~2.5% genuine mass in bin 0, about a quarter would pass the NM>5 min test anyway. That puts false-positive fills at ≈0.6 pp of 7.8 pp (~8%). This calculation belongs in the paper.
- **Strongest objection: NM itself is unreliable at LIRF.** In rows where BLOCK≠SCHED (|Δ| ≥ 2 min), **27%** still have NM AOBT_3 more than 5 min later than BLOCK. The base rate at EGLL, LEMD and LEBL is 10–12%. So "NM >5 min later" is not a fill-specific signal, and the repair is asymmetric: it is applied only where BLOCK equals a plan time. If NM replaces every row where NM is >5 min later, LIRF falls to 3.54 and every airport moves by −0.04 to −1.06.
- **Rank change robustness** (LIRF vs EGLL, annual, proxy on). It survives every correction I tried:

  | Correction | LIRF | EGLL |
  |---|---|---|
  | Drop the fills | 5.65 | 6.37 |
  | Drop all BLOCK=SCHED/EOBT rows | 5.98 | 6.42 |
  | Impute fills with the stand×runway median of non-copy flights (NM-free) | 5.42 | 6.35 |
  | NM-only taxi times | 2.22 | 6.97 |

  - It only becomes a **tie** if fills are assumed just 5 min too long: 6.37 vs 6.38.
  - ADS-B gives an independent size for the error. On LIRF upper-bound rows, BLOCK=SCHED with NM late (n = 48) sits 23.1 min before taxi start, against 9.5 min for normal rows. That is an overstatement of ≈13.5 min, about 80% of the NM-implied 17 min. Scaling the correction to 80% gives about 5.8, still below EGLL.
  - Day-bootstrap of the corrected EGLL−LIRF gap: **0.86 [0.66, 1.07]**.
  - Monthly: LIRF #1 goes 7 → 0 months, and EGLL is #1 in all 12. The "0–2" in the README is right.
  - The rank flip is also robust against PRU's published values (LIRF 6.735 − 1.29 < EGLL 6.505).
- **Circularity.** Partly circular but harmless. For fills, the NM taxi time averages 15.1 min, almost the same as non-fills (15.5). The "repair" therefore inserts a typical, largely table-derived value, and it behaves like median imputation (5.42).
- Must do:
  - Relabel 7.8% (SCHED) vs 9.2–9.4% (all plan times).
  - Present the NM-free imputation and the ADS-B-scaled correction as the primary estimates, with NM as one variant.
  - Report the 27% NM-late base rate at LIRF.
  - Give a CI for Δ.

## F3 – NM AOBT_3 is not a measurement. Verdict: holds for LEBL/LEMD/LSZH/EDDM; numbers depend on binning; the "table" attribution is not identified
- Recomputed share in the modal 2-min pair (cells stand×runway×month, n ≥ 20, minute **floor**):

  | Airport | NM share (theirs, rounded minutes) | NM share (mine, floor) |
  |---|---|---|
  | LEBL | 0.98 | 0.965 |
  | LEMD | 0.97 | 0.956 |
  | LSZH | 0.91 | 0.935 |
  | EDDM | 0.86 | 0.842 |
  | LTFM | 0.80 | **0.68** |
  | EDDF | 0.62 | 0.66 |
  | LFPG | 0.62 | **0.53** |
  | LIRF | 0.51 | 0.47 |
  | EHAM | 0.39 | 0.38 |
  | EGLL | 0.22 | 0.21 |

  - The airport-clock control is 0.21–0.37.
  - LEBL and LEMD are robust, including at n ≥ 100. LTFM and LFPG move by about 0.1 between rounding and floor (jitter interacting with bin edges), so report both.
- Within-cell correlations match theirs only after trimming |Δ| > 3 h. Pearson for LIRF is 0.43 untrimmed vs 0.145 trimmed; Spearman is 0.17 either way. Use Spearman.
- **Strongest objection:** concentration shows the value is *derived* (ATOT minus a near-constant), not *whose* table it is. These alternatives are not excluded:
  - the NM ETFMS default taxi time per runway configuration (this fits LTFM's runway-direction stripes);
  - airport-provided EXOT/taxi times sent in A-CDM DPI messages (LEBL, LEMD, LSZH, EDDM are all A-CDM airports);
  - NM back-calculating when no actual AOBT DPI arrives.
- LSZH's within-cell r = 0.36 together with 0.93 concentration points to a mixture, not a pure table.
- Must do: test the source of the constant. Compare the stripes with published ENV taxi times and with variable taxi time behaviour, and see whether the constant changes on the AIRAC cycle. Word the claim as "derived, source unidentified". Note that this behaviour may be known to NM operators, so cite NM ETFMS documentation.

## F4 – ADS-B physical reference. Verdict: holds with caveats for EHAM/LSZH/EDDM/LEBL; does not hold as stated for EGLL/LIRF/LEMD
- Matching logic is sound: callsign or flight number, take-off within ±120 s of MVT, best candidate by score. The detector is label-free.
- Bootstrap 95% CIs on the medians:

  | Airport | n | Median (s) | 95% CI |
  |---|---|---|---|
  | EHAM | 911 | +15 | [14, 16] |
  | LSZH | 234 | −21 | [−23, −19] |
  | EDDM | 641 | −30 | [−33, −28] |
  | LEBL | 381 | −22 | [−28, −17] |
  | EDDF | 158 | −27 | [−45, −13] |
  | LEMD | 65 | −91 | [−104, −81] |
  | EGLL | 116 | −153 | [−166, −133] |
  | LIRF | 40 | −105 | [−174, −68] |

- **Selection bias.** The precise pushback subset P covers only these shares of departures: EGLL 4.4%, LIRF 2.4%, LEMD 2.9%, EDDF 6.6%.
  - Transponder-on more than 60 s after BLOCK: EGLL 99.7%, LIRF 99.3%, LEMD 100%.
  - At those airports, P is the atypical subset with an early transponder. Half of EGLL's P rows have a stand dwell < 60 s (median offset −162).
  - LEMD's P flights taxi longer than the population (median 20.2 vs 16.2 min). LIRF's P subset has half the population's BLOCK=SCHED share (0.10 vs 0.18).
- **Stability claim is false for three airports.** "Stable across days and Jan/Jul" fails for:
  - EGLL: 109/116 rows from January (1 and 6 rows on the July days);
  - LIRF: 35/40 from one day, 07-09;
  - LEMD: July only.
  The day-level IQR at LIRF is 170 s and 1,109 s. With 2–4 days no day-clustered inference is possible.
- **Detector artefacts.** Sensitivity to the onset definition is small (≤ 20 s), but a common detector lag shifts every airport equally. Only the between-airport differences are identified, not the sign ("BLOCK before physical motion").
- **Statistical caveat on the "physical fill" evidence.** The README's "physically confirmed at LIRF" rests on **4** pushback rows. The upper-bound analysis (n = 100 SCHED rows, 21% with BLOCK > 30 min before taxi start) is stronger evidence and should be cited instead.
- **Missing link to the KPI.** Constant per-airport definitional offsets cancel in ATXOT (taxi − P10 of the same combo). F4 offsets matter for the KPI only through their dispersion and correlation with congestion. The paper should say this explicitly; otherwise readers will infer bias.
- Must do:
  - Drop or flag EGLL/LIRF/LEMD medians as indicative.
  - Ingest the 62 days before claiming stability.
  - Report selection rates and P-vs-population covariates.
  - Give CIs.
  - Remove the "stable" sentence for EGLL/LIRF/LEMD.

## Statistics overall
- None of the four findings currently reports a CI.
- 10 airports × multiple flag thresholds × 15 scenario classes × 3 modes is a large garden. Pre-specify the primary correction (I suggest NM-free imputation) and report the rest as sensitivity.

## Overall verdict
**Major revision.** The LIRF schedule-fill finding and the LIRF→EGLL rank flip are real and robust: every correction I tried keeps LIRF below EGLL, bar one tie case, and the bootstrap gap is 0.86 [0.66, 1.07] min. However:
- F1's 0.993 overstates the replication quality and depends on an un-validated de-icing proxy.
- The fill share is mislabelled (7.8% vs 9.4%).
- The NM-based repair rests on an NM clock that is >5 min late on 27% of normal LIRF rows.
- F4's results for EGLL/LIRF/LEMD come from tiny, selected, mostly single-season samples.

**Single weakest point:** F4 at EGLL/LIRF/LEMD, together with the claim that it is "stable across days and seasons". n = 40–116 drawn from 2–4% of departures with an early transponder, mostly from one season (one day for LIRF), with no day-level inference possible.
