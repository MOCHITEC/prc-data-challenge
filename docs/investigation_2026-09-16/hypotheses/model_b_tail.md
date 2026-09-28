# Hypothesis generator: tail / unmatched rows / loss (subagent output, verbatim, 2026-09-16)

[verified] = computed in the agent's session from data/work/feat_train.parquet, data/work/feat_rank.parquet, experiments/oof_*.parquet, outputs/*.parquet, git show. Nothing trained. [supposed] = inference.

## Headline facts (verified, not in docs/handoff.md)
- TAXITIME_SEC_mvt == MVT_TIME − BLOCK_TIME exactly for all 2,085,047 rows.
- Of the 119 unmatched rows with y ≥ 4h, 102 have |y − mvt_minus_sched| < 1,800 s; median residual 1 s. For those rows BLOCK_TIME == SCHED_TIME (±6 s): the label is literally MVT − SCHED, and mvt_minus_sched is a model-B feature present in ranking (null rate 0).
- 115 of the 119 are LIRF unmatched (1,488 training rows, 1,487 with AIRCRAFT_TYPE_mvt null, all IFR; median y 3,922 s; 29.5 % have y ≥ 2h; 48.6 % have |BLOCK−SCHED| ≤ 120 s). Sched-filled fraction given mms ≥ 2h: LIRF 0.633, all others ≤ 0.004.
- In OOF main-48c19bc830, 8 rows with y ≈ 86,400 + taxi ("day rollover") carry 52.7 % of unmatched SSE (≈25 % of total); 632 sched-filled rows carry 10.5 %; top-25 rows 76.3 %.
- Ranking: 5,290 unmatched; LIRF 383 (all actype-null), 181 with mms ≥ 2h, 43 ≥ 4h, 11 ≥ 12h (max 111,654). Training LIRF unmatched Jan+Jul 2025: 397 rows, 181 mms ≥ 2h, 11 ≥ 12h — same density.
- v4's model B used linear_tree (`git show 9f31cef:src/prc/train.py`: model B called _fit(..., cfg, ...); lgb_params() adds linear_tree; for_unmatched() did not exist). On ranking LIRF rows v4 predicted ≈ mms (108,552 for mms 111,654; 79,686 for 87,484) while v8 saturated at 40,267 / 34,552. RMS(v4 − v8) on unmatched ranking rows = 1,736 overall, 2,967 on the 1,721 rows with mms ≥ 2h, 469 elsewhere.
- A leave-one-month-out lookup table mean(y | ADEP × mms-bin) beats model B on unmatched OOF rows: 1,755 vs 2,296 (fold-honest forward 1,262 vs 2,109; jul 1,458 vs 2,009; jan 2,961 vs 3,223). Replacing only LIRF predictions with the table moves overall OOF 334.58 → 298.05.
- Forward fold model B stops at 44–51 rounds in every run; jan 175–538, jul 321–479, production 199–658. Forward's inner holdout is October = fewest extremes (4 rows ≥ 4h, max 25,558); production's holdout is December (6 rows ≥ 4h, max 87,605).
- Seed averaging: main-ab56551b5e (3 seeds) unmatched OOF 2,312.5 vs det_a (1 seed) 2,299.0. det_a = det_b = repro_v2 bit-identical.
- Code diff 9f31cef → a7a300d for model B: only force_row_wise, for_unmatched() (same 127/50/2000 via config), explicit bagging_seed/feature_fraction_seed. Nothing structural.
- Implied unmatched RMSE from LB with matched fixed at 218: v2 1,815 / v4 1,736 / v5 1,961 / v7 2,145 / v8 2,157 / leader 929.

## Table
| id | hypothesis | needs training |
|---|---|---|
| sched-filled-block | Extreme labels are BLOCK_TIME := SCHED_TIME records, so y = mvt_minus_sched exactly; the tail is deterministic given a feature the model already has. | no |
| rollover-86400 | ~8–15 rows/year have y = 86,400 + true taxi; ~half of unmatched SSE; predictable only as a mixture component. | no |
| lirf-regime | LIRF unmatched rows are a distinct data source (no aircraft type, sched-filled block); ranking carries the same density; the metric is decided by ~50 LIRF rows. | no |
| tree-identity-saturation | Model B saturates at ~35–40k because trees cannot represent y = mms and min_data_in_leaf 50 exceeds the number of LIRF rows with mms > 43,200 (29). | partly |
| linear-tree-modelB | v4's 37 s LB advantage over v7/v8 is mostly linear_tree in model B extrapolating along mms on ~2 ranking LIRF rows. | probe or run |
| mms-reparam-or-mixture | Model B should predict P(sched-filled|x)·mms + (1−P)·taxi; a 2-variable lookup already beats the GBDT. | yes (cheap) |
| false-alarm-non-lirf | Model B fires 15–20k predictions on large-mms rows at non-LIRF airports whose labels are normal (p_sf ≈ 0); ranking EHAM has 559 such rows. | no |
| early-stop-holdout-month | Forward fold's model B is a 45-round underfit because October has almost no extremes; screening on folds [forward] evaluates a model B unlike production's. | yes (small) |
| robust-loss-harmful | Huber/quantile/log/clipping will worsen RMSE here because the tail is predictable signal (clipping at 7,200 moves unmatched OOF 2,296 → 3,373). | clip no; loss yes |
| seed-avg-no-effect-B | Seed averaging does not move model B (2,312 vs 2,299); the v2→v8 lottery came from random-stream changes on a saturating model. | no |
| v5-verdict-confounded | The "255/20 cost 13 s" verdict compared v5 (new code) with v2 (old code); vs same-code v7/v8, v5 is 17–19 s better. | probe |
| shared-b-no-transfer | Training model B on all rows will not help the tail: the LIRF actype-null sched-filled regime does not exist among matched rows (119 of 122 rows y ≥ 4h are unmatched). | yes |
| cond-mean-vs-median | For LIRF & mms ≥ 12h, y ∈ {mms (62 %), ≈87k (38 %)}; RMSE-optimal is the mixture mean (~mms + 10k); a median loss would pick mms. | no |

## Per-item detail

### sched-filled-block
y ≥ 4h (n=119): residual y−mms quantiles [−6,365, −5, −2, 1, 4, 20,922, 86,402], 102/119 within 1,800 s; 2h ≤ y < 4h (n=362): 87.6 % within 1,800 s. None at non-LIRF except two LFPG rollovers. BLOCK seconds-of-minute within 6 s of a minute boundary for 100 % of LIRF rows (matched and unmatched) and 74 % of other unmatched rows; ranking mvt_minus_sched null rate 0.

### rollover-86400
8 rows carry 52.7 % of unmatched SSE (RMSE on them 53,927); e.g. LIRF y=87,480 / mms 56,520 / BLOCK−SCHED −30,960; LFPG y=84,240 / mms 1,740. MVT and BLOCK differ by 1 day + 13–30 min. Ranking has 11 LIRF rows with mms ≥ 12h, which in training are 62 % sched-filled (y = mms) and 38 % rollover (y ≈ 87k).

### lirf-regime
Training LIRF unmatched 1,487/1,488 actype-null, all FLIGHT_RULE "I"; ranking LIRF unmatched 383/383 actype-null, Jan 107 (52 mms ≥ 2h, 3 ≥ 12h), Jul 276 (129, 8); training Jan+Jul 2025: 60+337 rows, 24+157 mms ≥ 2h, 2+9 ≥ 12h.

### tree-identity-saturation
Ranking LIRF rows mms 47k–111k → v8 predicts 28–40k (one row 68k); ab_split (255/20) 50–71k incl. 71,322 for mms 47,705; OOF sched-filled rows with mms ≥ 2h have pred/mms median 0.83, p10–p90 [0.55, 1.10]. LIRF training rows with mms > 43,200 = 29 < min_data_in_leaf 50; in (14,400, 43,200] = 98. Training check: forward fold with unmatched_min_data_in_leaf 5, compare pred/mms on LIRF sched-filled OOF rows.

### linear-tree-modelB
RMS(v4−v8) = 2,967 on mms ≥ 2h rows vs 469 elsewhere; two LIRF rows mms 111,654 / 87,484: v4 108,552 / 79,686, v8 40,267 / 34,552. [supposed] if those two labels equal mms (training p = 0.62 for that bin), their SSE difference (≈7.8e9) is ~90 % of the total v4→v8 SSE gap ((343.80² − 305.03²)·344,841 ≈ 8.7e9). v4 predicted lower than v8/v5 on large-mms non-LIRF rows (EHAM 85,535 → 2,865 vs v5 8,626). Checks: (a) done; (b) forward-fold run with use_linear_tree true (~5 min), compare LIRF sched-filled pred/mms; (c) probe submission v4-matched + candidate-unmatched.

### mms-reparam-or-mixture
LOMO bin-mean by (ADEP × mms-bin) beats model B on unmatched rows on all three folds (forward 1,262 vs 2,109; jul 1,458 vs 2,009; jan 2,961 vs 3,223). Mixture with LOMO table P(sched-filled | ADEP, actype-null, mms-bin): 1,937 (soft) / 1,897 (hard at P ≥ 0.5) vs 2,296. Oracle "sched-filled → mms" alone: 2,172 (rollover rows dominate). Training version: model B target y − mms on LIRF rows, or classifier on 22,470 rows.

### false-alarm-non-lirf
Non-LIRF unmatched, mms in (14,400, 43,200]: 368 rows, sched-filled fraction 0.000; (43,200, ∞): 46 rows, 0.000. Top-25 SSE includes EDDF y=661 pred 16,880 (mms 89,185), EHAM y=549 pred 15,335 (mms 151,639), LFPG y=652 pred 15,812. Ranking EHAM 559 rows mms ≥ 2h, 148 ≥ 4h, 8 ≥ 12h; v5 predicted 7–10k on those, v8 2–7k, v4 0.3–5k.

### early-stop-holdout-month
Every forward-fold run logs rounds_unmatched 44–51; _inner_holdout_mask takes latest training month = October for forward; October 4 rows ≥ 4h, max 25,558 (lowest); production holdout December 6 rows ≥ 4h, max 87,605; production rounds 199/212/585/658. [supposed] at lr 0.05 a 45-round model cannot reach leaf means of 40k+ from base ~1,400. Check: forward-fold run with fixed unmatched rounds (e.g. 600) or inner_holdout_months 2.

### robust-loss-harmful
On oof_main-48c19bc830 unmatched rows: model 2,296; clip 3,600 → 3,568; 7,200 → 3,373; 14,400 → 3,070; 43,200 → 2,306; shrink 0.5·pred + 0.5·airport-median → 2,825; constant airport-median → 3,724; oracle airport-mean → 3,492. Rows y ≥ 4h: mean y 38,198, mean pred 25,800.

### seed-avg-no-effect-B
2,312.5 (3 seeds) vs 2,299.0 (1 seed); LIRF 7,253 vs 7,206. git diff 9f31cef a7a300d -- src/prc/train.py: model-B-relevant changes = force_row_wise, for_unmatched(), explicit bagging_seed/feature_fraction_seed per seed.

### v5-verdict-confounded
v5 325.22 vs v7 342.59 / v8 343.80 (same code family). Implied unmatched RMSE v5 1,961 vs v8 2,157. Caveat: v5 trained before the hour_utc fix (34808da), as was v4; v2 had no weather hence hour_utc present. v5 vs v8 on the 11 ranking LIRF mms ≥ 12h rows: v5 30–61k, v8 29–69k — nearly the same, so [supposed] the v5 gain over v8 is elsewhere.

### shared-b-no-transfer
Matched rows with |BLOCK−SCHED| ≤ 120 s are 11–26 % per airport but matched y ≥ 2h is 103 rows of 2.06M (47 in OOF, 0.9 % of matched SSE, y − proxy median 6,960); of 122 training rows y ≥ 4h, 119 unmatched; actype-null among matched LIRF: 0.

### cond-mean-vs-median
Bin (43,200, ∞) at LIRF: 29 rows, 62.1 % with y = mms; the rest are 87k rollover rows. Forward fold LIRF & mms ≥ 4h (n=10): mean y 48,907, model 26,296, bin-mean 44,405, mms 42,960. [supposed] for mms = 60k the mixture mean ≈ 70k gives RMSE ≈ 13k vs ≈ 16.6k for predicting mms and ≈ 25–50k for the current plateau.

## Not checked
- Whether linear_tree on model B over-extrapolates on non-LIRF large-mms rows under current code.
- Whether an mvt_minus_sched-monotone constraint helps ([supposed] wrong for non-LIRF).
- Label plausibility of 251 unmatched / 137 matched rows with y ≤ 0 (negligible SSE).

Key files: src/prc/train.py (_fit, _inner_holdout_mask, for_unmatched, cols_b — mvt_minus_sched and mvt_minus_sched_slog visible to model B), experiments/oof_main-48c19bc830.parquet, outputs/probe_unmatched_from_v4.parquet (v4's unmatched predictions), data/work/feat_rank.parquet.
