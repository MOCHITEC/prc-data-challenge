# Matched-lane big errors (2026-10-06): nothing predictable

7-month OOF (p = b + c_C). Matched rows with |e| > 600 are 15 737 rows and 27.0 % of SSE (|e| > 1800: 1 044 rows,
9.7 %). Mostly y > p (21.1 %), i.e. delayed flights whose BLOCK is 10–60+ min before AOBT_3. LIRF 10.9 %, LFPG 4.0,
LTFM 3.9, LEBL 2.6, EGLL 2.4, every other airport < 1 %. Disruption days are diffuse (the top 100 of 1 807
airport-days hold 9.7 %).

| Regime among \|e\| > 600 | rows | SSE share | verdict |
|---|---|---|---|
| LIRF schedule/LOBT copies | 1 725 | 6.7 % | handled by the production blend |
| Copies elsewhere (LEBL 921) | 1 452 | 2.1 % | coincidences on on-time flights; LEBL's real copies are seasonal and day-clustered, not predictable |
| AOBT_3 ± whole hours | 46 | 0.26 % | no spike |
| ADS-B agrees with AOBT_3 not BLOCK (label noise) | 1 069 | 1.6 % | unpredictable |
| AOBT_3 wrong, ADS-B near BLOCK | 7.8k–11.3k | small | already handled by the learned residual |

Leave-one-month-out copy blends toward LOBT/SCHED at LEBL, LEMD, LTFM, LFPG, EGLL: −0.5 to +0.1. Retuning the LIRF
blend weight: ≈ 0 on Jan+Jul. 2026 visible fields (ARR copies, ADS-B push vs AOBT_3, AOBT_3 − LOBT) look like 2025.
Closed. Scripts: `analysis/local_bigerr_{prep,anatomy,copy_lomo}.py`.
