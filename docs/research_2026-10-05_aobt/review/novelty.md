# Novelty and positioning review: AOBT audit paper (JOAS)

Reviewed: docs/research_2026-10-05_aobt/README.md. Web search done 2026-10-05.

## 1. Closest prior work and what each already covers

1. **PRU, *Additional taxi-out time performance indicator document*, Ed. 01.00, Mar 2023.** https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf. Gives the full method (P10 per stand×runway, 10-flight validity rule, de-icing exclusion). Sec. 4 says "the precision in the collection of the AOBT is key", and Sec. 8.3 lists the APDF checks (mapping, completeness, matching, "acceptability criteria" on OOOI). It has no check for schedule copying. This supports the authors' gap claim, but it also means F1 just follows a public recipe.
2. **Goens, Koelle, Spinielli, Bolić, Cook, Strohmeier, "The Open Performance Data Initiative…", SIDs 2024.** https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_100%20final.pdf. I confirmed that it compares only flight counts and runway-use counts with APDF, never timestamps. The authors' description is correct.
3. **Schultz et al., "A-CDM Lite…", SIDs 2019.** https://www.sesarju.eu/sites/default/files/documents/sid/2019/papers/SIDs_2019_paper_76.pdf. Off-block and other milestones from ADS-B at a small airport.
4. **Schultz, Rosenow, Olive, "Data-driven airport management enabled by operational milestones derived from ADS-B messages", J. Air Transport Management 99 (2022).** https://www.sciencedirect.com/science/article/abs/pii/S0969699721001459. The summary does not cite this. It is the main peer-reviewed reference for deriving off-block and taxi milestones from ADS-B and must be cited. I could not open the full text (403), so I could not check whether it validates against airport AOBT. Check before claiming "never compared".
5. **Olive, Waltert, Schultz, "Assessing Airport Surface Traffic Performance from Open Sources of Aviation Data", ATRDS 2025.** https://zenodo.org/records/20087716. Taxi-out KPIs at LSZH from open ADS-B, with no APDF comparison (agrees with the authors).
6. **Stloukal, Hád, Hospodka, "Analyzing the Duration of Aircraft Pushback Operations Using ADS-B Data", NTCA 2026 (Prague).** https://www.proceedings.com/content/086/086217webtoc.pdf. Detects pushback in ADS-B. Not cited. It is direct prior art for the physical reference used in F4.
7. **Hotle, Baszczewski, Gulding (FAA), "Using ASDE-X Surveillance for Taxi-Out Time Benchmarking and Delay Estimation", AIAA ATIO 2016, doi:10.2514/6.2016-3749.** https://vtechworks.lib.vt.edu/items/40dcfaee-5f35-4113-8284-38da65c90d43. Covers:
   - OOOI timestamps are 1-minute resolution and come from only part of the carriers.
   - Non-reporting carriers get *imputed* times.
   - Airline-reported wheels-off is offset from ASDE-X wheels-off, while radar-derived times are centred on zero.
   - Surveillance misses the non-movement-area part of taxi (about 2 to 6 min).
   - Unimpeded references shift by more than 5 min when terminal or stand is ignored.

   This is the US precedent for "reported times differ from surveillance, and some reported times are not measurements".
8. **Assaia, "Insight: Actual Off-Block Time Accuracy" (industry white paper, 2021 data).** https://www.assaia.com/resources/actual-off-block-time-accuracy. At one US airport (87 gates), ACARS AOBT differs from gate-camera AOBT by 180 s on average, and airlines differ by up to 6 min. This is not peer reviewed but is the closest published analogue of F4. Assaia also works with ADR at FCO (https://www.assaia.com/customer-stories/aeroporti-di-roma-reduces-departure-delays-by-6-minutes-with-assaia).
9. **EUROCONTROL Specification for A-CDM (Ed. 1.0, Jan 2025)** https://www.eurocontrol.int/sites/default/files/2025-01/eurocontrol-specification-for-acdm.pdf and **NM DPI Implementation Guide 2.700** https://www.eurocontrol.int/sites/default/files/2025-06/eurocontrol-dpi-impl-guide-2-700.pdf.pdf. Both are public:
   - The airport AOBT "may differ from the AO's OOOI times".
   - EXOT/VTT at CDM airports is a per stand×runway table.
   - CACD holds default taxi times per runway.

   This makes F3 expected from public documentation.
10. **Public PRC 2026 competitor repositories (not peer reviewed, but timestamped and public):**
    - RavindraTarunokusumo, PR #2, 2026-09-27: "At LIRF the long 'taxi-outs' are mostly a block-time-equals-schedule recording convention: block time sits at the scheduled time in 83 % of tail rows." https://github.com/RavindraTarunokusumo/prc-data-challenge-2026/pull/2
    - EnioAguiar: "84.7 % of the flights with y > 3 h satisfy |BLOCK − SCHED| ≤ 60 s." https://github.com/EnioAguiar/prc-taxiout-2026
    - ahmetabdullahgultekin: "faithful reimplementation" of the P10 stand×runway reference. https://github.com/ahmetabdullahgultekin/prc-taxiout-2026

    So the existence of schedule copying at LIRF and a reimplemented ATXOT reference are already public knowledge in the challenge community.
11. Context: PRU/FAA US–EU operational comparison 2015 (OOOI vs APDF data sources) https://www.eurocontrol.int/sites/default/files/content/documents/single-sky/pru/publications/other/us-eu-comparison-2015.pdf. PRB Italy monitoring report 2022 (LIRF ATXOT about 5 min, 2nd highest; ENAV says it cannot replicate PRU's data). https://www.sesperformance.eu/download/2022/PRB-Annual-Monitoring-Report_Italy_2022.pdf (download failed; quoted from the search snippet only).

I found no peer-reviewed work that (a) quantifies schedule copying in APDF and its effect on ATXOT, or (b) compares European airport AOBT with ADS-B pushback timing to the second across several airports. The Schultz 2022 JATM paper is the one I could not rule out.

## 2. Novelty per claim

**F1. KPI replication, r = 0.993: already known / low novelty (validation step only).** The challenge data is almost certainly APDF extracted by PRU, so agreement with PRU's own published values is close to circular. It confirms the provenance of the data, not anything new about the world. The method is public (ref. 1) and has been reimplemented (ref. 10). Keep it as a short sanity section, not a contribution. Also, "BLOCK is effectively APDF AOBT" is an inference [X]. Ask the organisers to confirm it; do not leave it as a deduction.

**F2. Schedule-copying bias at LIRF, −1.29 min (−19%), ranking flip: new in quantification, but not as a discovery.** The phenomenon is publicly documented by competitors (ref. 10) and is analogous to the ASPM imputation issue (ref. 7). New elements:
- the excess-over-adjacent-bins estimator, which separates copying from coincidence;
- propagation through the official KPI;
- the ranking change;
- the counter-intuitive result that dropping NM-matching rows is harmful.

This is the strongest claim, but it has an internal weakness. The correction substitutes NM AOBT, and F3 says NM AOBT at LIRF is partly table-derived. So the "corrected 5.50" is partly a modelled taxi time fed back into a KPI that measures deviation from a reference. A reviewer will call this circular. Options:
- bound the effect, e.g. by dropping FILL flights, or by sensitivity runs with NM, with drop, and with the P10 reference plus a median;
- validate against ADS-B (n = 40 at LIRF is too few; ingest the 62 days).

**F3. NM AOBT_3 is mostly table-derived: incremental.** The mechanism follows from public NM documentation (CACD default taxi times, VTT per stand×runway, EXOT in DPI; ref. 9), and the US analogue (ref. 7) is documented. The contribution is a per-airport quantification: concentration 0.22 to 0.98, near-zero within-cell correlation, the LTFM two-band structure, and 17% of EDDM AOBT_3 after taxi start. That is useful to researchers who treat NM/OPDI flight-list AOBT as measured. Present it as "users should know", not as a revelation. Note that "AOBT_3" is a dataset field name. Explain its NM provenance (A-DPI versus ATOT − taxi time), or a reviewer will say the authors misread a field designed as an estimate.

**F4. Airport-specific definitional offsets against ADS-B pushback: incremental to new (European first, globally a replication).** The finding that reported off-block differs from physical movement by airline or source, by roughly minutes, is established for the US through OOOI vs ASDE-X (ref. 7) and through vision-based work (ref. 8). What is genuinely new:
- the result is multi-airport, in Europe, from open data only, to the second;
- the offsets are stable across seasons (EHAM +15 s, EGLL −153 s);
- the A-CDM-lite "first-signal" proxy fails at EGLL, EDDF and LIRF.

There is a caveat on relevance. A constant per-airport offset cancels in ATXOT, because the actual time and the P10 reference share the same AOBT definition. The offsets matter for (i) comparing absolute taxi-out across airports, (ii) replacing APDF with ADS-B-derived KPIs (the OPDI ambition), and (iii) ML targets. State this explicitly, or reviewers will say F4 does not affect the SES KPI. The sample is also small (4 days; LIRF n = 40, LEMD n = 65, EGLL n = 116), and detection depends on ADS-B coverage, with pushback often happening with the transponder off. That creates selection bias: the authors themselves report that the transponder comes on more than 60 s after AOBT in 92 to 99.7% of rows at EGLL, EDDF and LIRF. This raises the question of how pushback onset was observed for those flights at all. The answer must be explained clearly.

## 3. Suggested reframing

Title idea: "Auditing reported off-block times in European airport performance data with open surveillance: schedule copying, modelled timestamps and definitional offsets."

- Make the contribution an **open, reproducible AOBT data-quality audit method** with three directional tests:
  1. copying detection by excess over adjacent bins;
  2. a test for table-derived timestamps by concentration within cells;
  3. physical-reference offsets from ADS-B.

  Then propose these as additions to the APDF "acceptability criteria" (ref. 1, Sec. 8.3).
- Lead with the KPI consequence (F2), shown as a sensitivity band rather than a single corrected number.
- Explicitly position against the US OOOI/ASDE-X literature (refs. 7, 8) and Schultz 2022 / Stloukal 2026. Say "first European, multi-airport, open-data replication and extension", and drop "never done".
- Cite the competitor repos honestly as informal prior observation of LIRF copying. Hiding them is a real risk, because JOAS reviewers will likely include PRC/OSN people who have read them.
- Expand the ADS-B data to the 62 days before submission. Without that, F4 rests on n = 40 to 116 at the key airports.

## 4. Risks a JOAS reviewer will raise

1. **Data licensing and open-data statement.** JOAS requires open data. The challenge data's licence and redistribution terms are unclear: the organisers have not released it, and the rows are held only locally. If the data cannot be shared, the paper may fail JOAS's open-data policy. Get written confirmation from PRU/OSN on publication and redistribution before submission. Identifying LIRF/ADR data-quality problems from data provided under challenge terms may also need organiser consent.
2. **Reproducibility.** Scripts write to a scratch path, and the de-icing proxy is built from METAR. The 4 days of adsb.lol data (ODbL, which needs attribution and share-alike on derived databases) must be archived with a DOI (Zenodo), or the exact pull scripts must be given. Possible anonymisation jitter or other organiser processing is unresolved.
3. **Circularity** (F1 is validated against its own source; the F2 correction relies on NM, which F3 discredits).
4. **Minute truncation:** EDDM's −30 s offset is explained by truncation; the authors must do the same truncation analysis for every airport before calling offsets "definitional".
5. **ADS-B selection bias** (transponder-off pushbacks, receiver coverage at EGLL/LIRF).
6. **Fairness to named operators:** naming ADR/LIRF and LEBL as having "distorted" official values needs careful wording, and ideally a right of reply or confirmation from PRU.

## Bottom line

The paper is publishable in JOAS as a data-quality and audit contribution, but its novelty is modest and must be stated honestly. F1 is not a contribution. F3 is expected from the documentation. F4 replicates a known US phenomenon in Europe with new open-data detail. F2's KPI-impact quantification is the real core, and it is currently undermined by its reliance on NM times.
