# EnioAguiar/prc-taxiout-2026 (team `outgoing-boat`): deep read, 2026-10-05

Clone: `scratchpad/repos/src/enio` (254 commits, last 2026-10-03 "trabalho encerrado"). Docs are in Portuguese. Main sources: `README.md`, `submissions.jsonl`, `docs/diario.md`, `docs/research/*`, `docs/esteira.md`.
**Licence: GPLv3** (LICENSE + NOTICE). Third-party data: adsb.lol (ODbL), EUROCONTROL ansperformance CSVs, OPDI, IEM METAR, and X-Plane `apt.dat` (GPL). Read for ideas only. No code was copied.
**Standing:** board rank 31, best **243.9532** (v37). The 244.89 figure is their v33. We are ahead of them (242.59), so anything we take from them is a marginal gain.

## 1. LB history (official RMSE, change, delta)

| v | change | LB | Δ |
|---|---|---|---|
| 1 | GBDT, train without y≥3h, pred capped at 3h | 514.53 | — |
| 2 | **keep outliers**, fix a µs→s unit bug in congestion windows | 384.74 | **−129.8** |
| 3 | two-stage: classifier P(\|BLOCK−SCHED\|≤60s) + L2 regressor on normal rows, mixed by **expectation** p·(MVT−SCHED)+(1−p)·ŷ | 338.67 | **−46.1** |
| 4 | per-airport line y=a+b·(MVT−SCHED) on all unmatched rows | 337.18 | −1.5 (sim promised −42: 10 LIRF rows) |
| 5 | line only where MVT−SCHED>6h | 330.95 | −6.2 |
| 6 | adsb.lol features (observed off-block, gs0, lat0/lon0) | 314.76 | **−16.2** |
| 10 | *diagnostic:* v6 + LOBT-window projection on its 117 rows only | 284.17 | **−30.6** from 117 rows |
| 9 | cross-fitted residual corrector (2-month out-of-block base) + LOBT window | 275.90 | **−38.9** (window −30.6, corrector −8.3) |
| 11 | corrector + arrival taxi-in and neighbour MVT−AOBT_3 (before *and* after) | 266.81 | **−9.1** |
| 12 | corrector = mean(global LGB, per-airport LGB, CatBoost) | 264.74 | −2.1 |
| 13 | Rome post-rule (LIRF unmatched, MVT−SCHED in (15h,30h] → a+0.38·ms; 4 rows) | 264.35 | −0.4 |
| 14 | drop adsb_lat0/lon0 (drift) | 266.28 | **+1.54 (worse; sim had promised −2.1)** |
| 17 | regressor target clipped at 2h + airline copy rate + EUROCONTROL daily series + OPDI ground time | 256.86 | **−7.5** |
| 19 | METAR, stand rotation, NM consistency | 253.95 | −2.9 |
| 21 | ctx features also in base | 252.34 | −1.6 |
| 27 | `dist_plano`: distance from the prediction to each planned clock | 251.10 | −1.24 |
| 28 | stand prefix (apron) category + 500 corrector rounds | 249.16 | −1.94 |
| 29 | per-airport base regressor + cell stats (P50/P90/sd/n by apt×stand×rwy) | 247.83 | −1.33 |
| 30 | X-Plane taxi-path distance in corrector | 247.76 | −0.07 |
| 32 | 2nd corrector member (queue + "surface at predicted push") | 247.11 | −0.65 |
| 33 | METAR/rotation/NM block moved into base, correctors refit | 244.89 | −2.21 |
| 34 | gate retention (overdue vs EOBT at AOBT_3, active on runway, 15-min mean AOBT_3−SCHED, EWMA rate) | 244.18 | −0.71 |
| 35 | corrector neither trains on nor corrects rule-served rows | 244.13 | −0.05 |
| 36 | 3rd member (runway config, wake-weighted queue) | 243.98 | −0.16 |
| 37 | 4th member on a 2nd base (`cls-peso quad`) | 243.95 | −0.02 (sim −0.77) |

The large moves came from data findings: keeping outliers, the expectation mixture, ADS-B, the LOBT window and the cross-fitted stacker. Everything after v21 was worth 2 s or less per submission. Ablations of each late sub-block (esteira 163–196) all sit within noise (+0.02…+0.3).

## 2–3. What they built, and whether we have it

| Item | Ours? |
|---|---|
| LOBT window ±3606 s projection | **have** (`lobt_bound`, same constant). Their v10 confirms it holds on 2026, but v10 was a one-row-set diagnostic submission, which is close to LB probing. Don't copy the method. |
| Expectation mixture on SCHED-copy (all rows) | different design: we anchor on AOBT_3 plus the LIRF reparam. Their multiclass "which clock was copied" version lost 4 s. Their LB is not better than ours, so no change. |
| Cross-fitted residual corrector, 2-month blocks | have (learned residuals, OOF) |
| Arrival taxi-in, neighbour MVT−AOBT_3 | have (`nbr_*`, `arr_taxi_in_*`, L2 forward, L5 tx_*) |
| METAR, OPDI turnaround, per-airport member, CatBoost in mix | have |
| Target clip 2h for regressor | have (7200 clips) |
| AOBT_3-anchored queue counts | have (L3 `qp_*`) |
| **Regime-split transfer gate (body share)** | **no** (we have tail-removed CI and tail report, but not a backtested additive body/tail split) |
| **Single-column adversarial AUC for engineered aggregates** | partial (`adversarial.py` exists; not used as a per-feature gate) |
| **Stand prefix (apron) category** | **no** (we use exact STAND_mvt + OSM + new-stand dropout) |
| **dist_plano in residual model** | probably partial: the clocks are features, but pred-minus-clock is not |
| **Surface counts at *predicted* push (MVT_j − pred_j)** | **no** (ours use AOBT_3, so they are missing on unmatched rows) |
| **Overdue-vs-EOBT_1 at own AOBT_3** | partial (`bk_sched_backlog_60m` is SCHED-based at take-off) |
| **EUROCONTROL daily series (ATFM regulated/off-slot share, pre-departure delay)** | **no** (surveyed 09-29, "not used") |
| Rule rows excluded from residual training | unknown. Check the overlays and the 4-month residual. |
| X-Plane apt.dat taxi distance | no. Worth −0.07 LB for them, GPL data. Skip. |
| Pseudo-labelling 2026 rows (weight 0.3, body only) | no. Never measured on the LB. Skip. |
| OpenFE-style auto FE | no. 1 of 755 candidates passed and its gain was noise. Skip. |

## 4. Places where their results contradict ours or deserve attention

- **Tail gains anti-predict the LB.** Over 6 consecutive submissions, the share of the simulated gain from y≤3600 s ("body") predicts the official Δ with Pearson 0.985, ratio 1.03, MAE 0.15 s (LOO 0.22 s). The share from y>3600 s has Pearson **−0.86**. Full RMSE: 0.44 s MAE. `sem_loteria` (11 flights dropped): 0.36 s. Importance-weighting by an adversarial 2025/2026 model (AUC 0.866) made the prediction worse (r 0.03). This bears directly on our L5 problem (CV −0.32, LB +0.12). (`docs/research/2026-10-03-transferencia.md`)
- **Within-frame group aggregates fingerprint the frame.** Mean/sd per (type×apt, apt×stand…) computed inside a frame gives single-column adversarial AUC 0.91–0.95, from source columns at 0.50–0.55. Centring does not fix it. Only rank-within-group survives (AUC 0.50). Our L5 `tx_*` ratios (window mean / training-side airport mean) and any per-frame tables are suspects. (`2026-10-03-fe-auto.md` §4)
- **Pruning features for drift backfired.** Dropping adsb_lat0/lon0 was predicted −2.1 s in sim and came back +1.54 s on the LB. Use adversarial AUC to gate *new* columns, not to prune existing ones.
- **ADS-B coverage shift.** Coverage at EGLL/LEMD is 79%/67% of rows in 2026 against 14%/2% in their 2025 holdout. Simulating 2026 coverage moved their score by ≤0.6 s. We should check that our learned ADS-B residual handles the higher 2026 coverage.
- **Exact stand hurt; apron prefix helped.** In their corrector, the exact stand category made things worse and the prefix did not. We feed the exact stand.
- Things that failed for them (keep in mind, no action): residual target on MVT−AOBT_3 (−0.2 s, tie; this matches our finding that the plain-y blend helps), seed averaging (0.2–0.3 s), XGBoost, isotonic calibration of p, bias correction from published monthly means (worse), T=D−G "rival prediction" fed to the corrector at LIRF (−1.48 s). Our LIRF reparam is a target reparametrisation, not a feature, so this does not contradict it.
- Calibration: their official/sim ratio was a stable 0.82–0.87. The simulation was a 2025 Jan+Jul holdout built the way the organisers build the ranking set.

## 5. Top 8 transferable ideas (ranked)

| # | Idea | Evidence | Expected for us | Effort | Risk |
|---|---|---|---|---|---|
| 1 | **Body/tail decomposition of every paired gain**: RMSE_b−RMSE_n = Σd_i/(N(RMSE_b+RMSE_n)), restricted to y≤3600. Gate on day-bootstrap CI>0 for the body plus top-20 rows < 200% of the body gain. **Backtest on our ~37 submissions first.** | `transferencia.md`. Caught v37 (sim −0.77, LB −0.02). 6/6 correct on their history. | Better calls on the last ~6 days of submissions. Avoids another L5. | 0.5 day, analysis only | none |
| 2 | **Single-column adversarial AUC gate (2025 holdout vs 2026 ranking)** for new or engineered columns. Replace frame-level means with rank-within-group. | `fe-auto.md` §4: AUC 0.91–0.95 vs 0.50–0.55 | Explains or avoids CV≠LB features (check L5 `tx_*`) | low | low (gate new columns only, given v14) |
| 3 | **Stand prefix (apron letter / first digit) × airport as a category**, alongside or instead of the exact stand. Gives unseen 2026 stands (EDDF T3, EDDM 103–108) a known parent. | v28 −1.94 LB (confounded with rounds). Alone +0.90 s sim (CI +0.32…+1.64). Rome diag −1.4 s over 3 seeds. | 0.1–0.5 s. Stacks with stand-dropout. | low | low |
| 4 | **`dist_plano`**: pred−(MVT−clock) for SCHED/LOBT/IOBT/EOBT/AOBT_3, min distance, count of clocks >1h away, distance to LOBT-window edges. Input to our learned-residual stage. | v27 −1.24 LB | 0.2–1 s if our residual model lacks it | low-med | low |
| 5 | **Surface counts at predicted push**: o_j = MVT_j − pred_j for all departures (with OOF preds on train). Count departures taxiing at my o_i, take-offs during my taxi, arrivals taxiing in. Covers unmatched rows, where our AOBT_3-based `qp_*` is null. | v32 −0.65 LB as an added member | 0.2–0.6 s, mostly the unmatched lane | med (two-pass) | med (OOF discipline) |
| 6 | **Exclude rule- and overlay-served rows from residual/corrector training**, and leave their predictions untouched. | v35 −0.05 LB | ≤0.1 s, but free | trivial | none |
| 7 | **Gate-retention block at own AOBT_3**: #{EOBT_1 ≤ t < AOBT_3} per runway and airport (only EOBT≤AOBT, wait <2h), plus 15-min mean AOBT_3−SCHED of recent pushes and an EWMA departure rate (half-life 10 min). | v34 −0.71 LB. Their sub-ablation is in the noise. | 0–0.3 s | low | low |
| 8 | **EUROCONTROL daily series** per airport-day: ATFM regulated share, off-slot share, pre-departure delay min/flight (all and ATC). May catch Jan-2026 disruption days. Organiser called ansperformance data open. Document it. | cheap screen −0.8 s (LTFM −2.3). Bundled in v17 −7.5. Later ablation is noise. | 0–0.5 s | low-med | low (licence attribution) |

Not recommended: X-Plane distances (−0.07, GPL data), pseudo-labels (unmeasured), seeds, XGBoost, multiclass copy mixture, a second base for diversity (−0.02), diagnostic-submission tactics (v10/v14 used the LB as an oracle, which our rules forbid).
