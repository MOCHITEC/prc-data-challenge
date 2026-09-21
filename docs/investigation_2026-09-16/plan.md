# Investigation plan — consolidated hypotheses (2026-09-16)

Consolidation of four independent hypothesis reports (49 candidate items) into one deduplicated
list. This file contains no recommendations, no ranking by importance and no new hypotheses;
every number is copied as written in its source. Interpretation stays with the source reports.

Sources (all in `docs/investigation_2026-09-16/hypotheses/`):

| short | file | items | agent scope |
|---|---|---|---|
| E | `external.md` | H1–H12 (12) | official rules, leaderboard, public 2026 repos, past editions, literature |
| P | `pipeline.md` | 1–12 (12) | pipeline / evaluation / reproducibility |
| D | `data_target.md` | 11 hypotheses + `checked-not-a-cause` (12) | data / labels / target / leakage / shift |
| M | `model_b_tail.md` | 13 | tail / unmatched rows / loss |
| H | `docs/handoff.md` | — | verified baseline facts (Japanese) |

Baseline (H): team resilient-hedgehog, best LB 305.03 (v4, 09-11); current-code builds 342.59 (v7) /
343.80 (v8); probe v9 (v8 matched + v4 unmatched) 306.46. Leader youthful-giraffe 245.02 (H; E's
2026-09-16 pull of the leaderboard shows vigorous-whistle 243.58, enthusiastic-daisy 244.83,
youthful-giraffe 245.02 — see §4). Model A (NM-matched): 2,062,577 training rows, OOF RMSE 217.7,
339,551 ranking rows. Model B (unmatched): 22,470 training rows, OOF RMSE 1,864.9, 5,290 ranking rows
(1.53 %). 48.1 % of total squared error sits on model B's side.

## 0. Definitions used below (read §4 before comparing numbers across items)

- `y` = `TAXITIME_SEC_mvt` = `MVT_TIME − BLOCK_TIME` (identity verified on all 2,085,047 training rows by D and M).
- `proxy` = `MVT_TIME − AOBT_3` (model A target decomposition, H §2).
- `ms` (D) = `mms` (M) = `MVT_TIME − SCHED_TIME` = feature `mvt_minus_sched` (visible to model B, ranking null rate 0).
- "unmatched" has two definitions:
  - **routing definition** (H, P, M): `mvt_minus_aobt3` / `AOBT_3` null → sent to model B → **22,470** training / **5,290** ranking rows;
  - **D's definition**: `FLIGHT_ID_mvt` null → **22,424** / **5,283** rows (the difference is 46 / 7 matched rows with null AOBT_3, which the routing rule also sends to model B).
- "fill" / "sched-filled" has several thresholds: D `fill = |BLOCK_TIME − SCHED_TIME| ≤ 10 s`; M uses `BLOCK_TIME == SCHED_TIME (±6 s)`, `|BLOCK−SCHED| ≤ 120 s`, and `|y − mms| < 1,800 s` in different places; D also uses `|y − ms| ≤ 60 s`.
- Thresholds in seconds: 2 h = 7,200; 4 h = 14,400; 12 h = 43,200; 1 day = 86,400.
- OOF file used by D and M: `experiments/oof_main-48c19bc830.parquet` (672,428 rows, RMSE 334.58; unmatched-row RMSE 2,296 per M). H's figures (217.7 / 1,864.9; 2-model split 218.5 / 1,875.3) come from earlier OOF runs, not this file.
- D's scripts: `docs/investigation_2026-09-16/scripts/q1.py … q5.py` (1–3 min each, `POLARS_MAX_THREADS=2`, no training). Sub-labels like "q2b C1c" refer to sections of those scripts.
- Flags: `no-training` / `model-B-fit (~1–2 min)` / `forward-fold run (~5 min)` / `full run (~20 min)` / `probe submission`.

---

## 1. Investigation items (35, grouped by theme; order is not a ranking)

### Theme 1 — label mechanism and the unmatched tail

#### 1. `sched-filled-block`
- **Sources:** D `lirf-sched-filled-block`; M `sched-filled-block`.
- **Hypothesis:** At LIRF, when no real off-block time exists, `BLOCK_TIME` is filled with `SCHED_TIME`, so the label becomes `y = MVT − SCHED` (= feature `mvt_minus_sched`); the extreme tail is this artefact, not taxiing, and is deterministic given a feature model B already has.
- **Verified:**
  - `TAXITIME_SEC_mvt == MVT_TIME − BLOCK_TIME` exactly for all 2,085,047 rows (M); within 1 s on 100 %, no null labels (D).
  - Label bucket composition (2,085,047 training departures): 2–4 h: 462 rows, 78 % unmatched, 71 % fill; 4–20 h: 106 rows, 98 % unmatched, 94 % fill; > 20 h: 16 rows, 12.5 % fill, 75 % BLOCK > 1 h before SCHED (D).
  - Of 481 unmatched rows with y ≥ 2 h, 82.5 % have |y − ms| ≤ 60 s. Of 103 matched rows with y ≥ 2 h, 34 % (D).
  - y ≥ 4 h unmatched (n=119): residual y − mms quantiles [−6,365, −5, −2, 1, 4, 20,922, 86,402]; 102/119 within 1,800 s; median residual 1 s; for those rows BLOCK_TIME == SCHED_TIME (±6 s). 2 h ≤ y < 4 h (n=362): 87.6 % within 1,800 s. None at non-LIRF except two LFPG rollovers (M).
  - 115 of the 119 are LIRF unmatched (M).
  - Fill rate LIRF unmatched 48.4 % (filled rows: y median 7,562, 55 % ≥ 2 h); LIRF matched 17.8 %; all other airports 0–5.7 % (D). 48.6 % of the 1,488 LIRF unmatched training rows have |BLOCK−SCHED| ≤ 120 s (M).
  - 2–20 h ms band: LIRF unmatched 622 rows, fill 64 %, y≈ms 64 %, y median 8,489; every other airport × group has fill 0.00–0.01 (D). Sched-filled fraction given mms ≥ 2 h: LIRF 0.633, all others ≤ 0.004 (M).
  - BLOCK seconds-of-minute within 6 s of a minute boundary for 100 % of LIRF rows (matched and unmatched) and 74 % of other unmatched rows (M).
  - `mvt_minus_sched` and `mvt_minus_sched_slog` are in `cols_b` (src/prc/train.py); ranking `mvt_minus_sched` null rate 0 (M).
  - OOF main-48c19bc830: 632 sched-filled rows carry 10.5 % of unmatched SSE (M).
  - Ranking: LIRF unmatched ms ≥ 2 h = 181 rows; `outputs/main.parquet` predicts median 7,041 for them (86 predicted ≥ 2 h) (D).
- **Unverified:** LB effect (ranking labels unknown); whether ranking LIRF rows are filled at the training rate.
- **Procedure:** D scripts q2b C1c/C1d, q4 I/I2/I3 on `data/work/feat_train.parquet` / `feat_rank.parquet`; columns `BLOCK_TIME`, `SCHED_TIME`, `MVT_TIME`, `TAXITIME_SEC_mvt`, `ADEP`, `FLIGHT_ID_mvt`, `mvt_minus_sched`; fill flag per definition in §0; tabulate by ADEP × matched/unmatched × ms band.
- **Flag:** `no-training`.

#### 2. `block-day-rollover`
- **Sources:** D `block-day-rollover`; M `rollover-86400`.
- **Hypothesis:** A handful of labels per year are `BLOCK_TIME` with the date off by one day (label = true taxi + 86,400); these rows dominate SSE and are predictable only as a mixture component.
- **Verified:**
  - 16 training rows y > 20 h; y − 86,400 = 602…1,992 s for 13 of 16 (e.g. AMX071: SCHED 06-24 21:15, BLOCK 06-24 15:29:54, MVT 06-25 16:03, y 88,392). Training: 15 rows over 12 months (D). All training rows y ≥ 43,200: 34 of 2,085,047 (P).
  - MVT and BLOCK differ by 1 day + 13–30 min; e.g. LIRF y=87,480 / mms 56,520 / BLOCK−SCHED −30,960; LFPG y=84,240 / mms 1,740 (M).
  - OOF main-48c19bc830: 10 rows y > 20 h = 34.7 % of OOF SSE (RMSE within group 51,121, pred median 37,131) (D). 8 rows with y ≈ 86,400 + taxi carry 52.7 % of unmatched SSE (≈25 % of total), RMSE on them 53,927; top-25 rows 76.3 % of unmatched SSE (M). H §3: y ≥ ~1 day, 10 rows, 24.1 % of OOF SSE.
  - Training ms bands: 20–28 h unmatched 23 rows (17.4 % y ≥ 2 h), > 28 h unmatched 3 (D).
  - Ranking: 6 + 1 unmatched and 36 + 6 matched rows in the 20–28 h / > 28 h ms bands (D). Ranking has 11 LIRF rows with mms ≥ 12 h, which in training are 62 % sched-filled (y = mms) and 38 % rollover (y ≈ 87k) (M).
- **Unverified:** number of rollover rows in the ranking set (needs labels); whether any feature separates rollover from sched-filled.
- **Procedure:** D scripts q2 C2/C2b, q5 N. On `feat_train`: rows y > 72,000; compute y − 86,400, BLOCK − SCHED, date(MVT) − date(BLOCK). On `feat_rank`: rows by ms band 20–28 h / > 28 h, split matched/unmatched.
- **Flag:** `no-training`.

#### 3. `lirf-regime-tail-count`
- **Sources:** M `lirf-regime`; P 3 `lb-decided-by-20-lirf-rows`; D `tail-count-per-set`; E H1.
- **Hypothesis:** LIRF unmatched rows are a distinct data source (no aircraft type, sched-filled block) present at the same density in the ranking set; the LB score, the differences between submissions, and the CV/LB disagreement are decided by a few dozen LIRF rows whose count varies per evaluation set.
- **Verified:**
  - Training LIRF unmatched 1,488 rows, 1,487 actype-null, all FLIGHT_RULE "I"; median y 3,922 s; 29.5 % y ≥ 2 h (M). AIRCRAFT_TYPE_mvt null on all 627 LIRF unmatched ms ≥ 2 h rows; runway 25 in 554/627 (D).
  - Ranking LIRF unmatched 383 (383/383 actype-null): Jan 107 (52 mms ≥ 2 h, 3 ≥ 12 h), Jul 276 (129, 8); 181 mms ≥ 2 h, 43 ≥ 4 h, 11 ≥ 12 h (max 111,654). Training Jan+Jul 2025: 60+337 rows, 24+157 mms ≥ 2 h, 2+9 ≥ 12 h (397 / 181 / 11 — "same density") (M). D: ranking Jan 107 / Jul 276, 181 with ms ≥ 2 h (median ms 10,135).
  - LIRF unmatched per training month 52–337 (Jul 2025 = 337); fill rate 32–76 %; y ≥ 2 h share 22–48 % (D).
  - OOF: LIRF unmatched 537 rows = 36.8 % SSE; LFPG unmatched 1,562 rows = 15.5 % (RMSE 2,731); other unmatched airports ≤ 1.5 % each (D).
  - Submission files, 5,290 unmatched rows, joined on MVT_ID_mvt to submitting.parquet (P):

    ```
    UNMATCHED 5,290 rows          med    mean   p99     max    n>7200 n>14400 n>43200
    v2  (old code, LB 312.02)    1062   1612  10256   68253   104    33      2
    up_v3 (old, LB 306.29)       1092   1675   9359  107724    82    31     16
    up_v4 (old, LB 305.03)       1112   1702   9457  108552    83    31     18
    v5  (LB 325.22)              1060   1559   9697   61356   106    32      2
    v7  (LB 342.59)              1082   1608  10169   66693   100    33      2
    v8  (LB 343.80)              1066   1608  10184   68581   101    35      2
    ```
  - RMS(v8 − v4) on unmatched = 1735.5 s; top-20 rows carry 73.2 % of Σgap²; all 20 are LIRF (v4 40–108 k vs v8 30–40 k). MATCHED 339,551 rows: RMS(v8 − v4) = 73.8 s → LB difference 1.43 s (v9 vs v4) (P). M: RMS(v4 − v8) on unmatched ranking rows = 1,736 overall, 2,967 on the 1,721 rows with mms ≥ 2 h, 469 elsewhere.
  - Submissions predict ≥ 2 h on 103–130 ranking rows (training frequency implies ≈97), ≥ 12 h on 3–19 (implies ≈6) (D).
  - Implied unmatched RMSE from LB with matched fixed at 218: v2 1,815 / v4 1,736 / v5 1,961 / v7 2,145 / v8 2,157 / leader 929 (M).
  - External comparison: OpenAir matched RMSE plateau ≈236 at LB 277 vs ours 217.7 at LB 305 (E H1).
  - H §3: unmatched RMSE 1,864.9 drops to 1,044 without the worst 20 rows and to 177.9 without the worst 2,000 (matched: 193.1).
- **Unverified:** true labels of the ~20 ranking LIRF rows — P supposes ≈ 86,400 s (day rollover), M supposes = mms with training p = 0.62; needs 2026 labels. E H1's "SSE share of unmatched rows by airport on OOF" is partly covered by D's 36.8 % / 15.5 % figures.
- **Procedure:** join `outputs/*.parquet` to `submitting.parquet` on `MVT_ID_mvt`; per-row gap² ranking between builds; SSE share by ADEP × matched/unmatched on `experiments/oof_main-48c19bc830.parquet`; D scripts q4 J, q2b C7, q5 P. `outputs/probe_unmatched_from_v4.parquet` holds v4's unmatched predictions (M).
- **Flag:** `no-training`.

#### 4. `lirf-matched-fill`
- **Sources:** D `lirf-matched-fill`.
- **Hypothesis:** LIRF matched rows also have sched-filled BLOCK; when AOBT_3 is far from SCHED the residual y − proxy = AOBT_3 − SCHED with large sd, making LIRF the worst matched airport.
- **Verified:** LIRF matched fill 17.8 % (by month 14.5–20.6 %); |AOBT_3 − SCHED| ≤ 600 s: fill 26.6 %, residual sd 253; > 600 s: fill 8.4 % (≈6,460 rows), residual sd 1,138, y sd 1,203. Fill by operator (hashed, n>2,000): 54 %, 25 %, 22 %, 19 %, 14 %, 11 %, ≤5.6 % rest; by segment Cargo 47 %, Non-Scheduled 45 %, Lowcost 22 %, Mainline 14 %. Matched |BLOCK − AOBT_3| > 1 h at LIRF: 457, y median 5,274, proxy median 1,025, 80 % filled. OOF LIRF matched RMSE 347 (next LFPG 246, EHAM 162), 7.9 % SSE; filled ∧ far subset (1,856 OOF rows) 3.0 % SSE, RMSE 1,106 (D). actype-null among matched LIRF: 0 (M).
- **Unverified:** LB effect; whether the fill state is recoverable for matched rows from available columns.
- **Procedure:** D scripts q2b C1d/C4b, q4 J, q5 N/N2; `feat_train` LIRF matched rows: fill flag, |AOBT_3 − SCHED| bands, sd of y − proxy; OOF residual by airport.
- **Flag:** `no-training`.

#### 5. `matched-proxy-outliers`
- **Sources:** D `matched-proxy-outliers`.
- **Hypothesis:** A set of matched rows has |BLOCK − AOBT_3| > 1 h (LTFM: stale early AOBT_3; EGLL/LFPG: BLOCK wrong), producing large errors under the proxy decomposition; some ranking matched rows are predicted ≥ 2 h.
- **Verified:** |BLOCK − AOBT_3| > 1 h on 1,808 matched rows; > 30 min 10,554; > 10 min 6.8 %. LTFM 657 rows: proxy median 4,692, y median 850. EGLL 296 / LFPG 186 / EDDM 45: y median 5,035 / 3,328 / 3,722, proxy 600 / 658 / 119. Matched |proxy − y| > 1 h: 689 OOF rows, 2.5 % SSE. Ranking matched predicted ≥ 2 h: EGLL 7, LIRF 7, EHAM 6, LFPG 2 (22). Ranking matched AOBT_3 null: 7 (D). Matched y ≥ 2 h is 103 rows of 2.06M (47 in OOF, 0.9 % of matched SSE, y − proxy median 6,960) (M).
- **Unverified:** whether the 22 ranking rows are the same kind; LB effect.
- **Procedure:** D scripts q2b C4b, q5 N/Q.
- **Flag:** `no-training`.

#### 6. `return-to-stand-twins`
- **Sources:** D `return-to-stand-twins`.
- **Hypothesis:** Departures recorded twice with the same (ADEP, FLIGHT, SCHED) and take-off within 2 min; the matched twin carries the earlier pushback's BLOCK, so its label is inflated.
- **Verified:** 223 pairs in training (450 rows), 129 mixed; take-offs within 120 s in 68–89 %; matched twin y median 3,170–3,834 vs unmatched 966. Ranking 84 rows in such groups (LFPG 38, LTFM 28, EDDF 10, LSZH 8). No MVT_ID overlap ranking/training; 4 FLIGHT_ID overlaps. Upper bound LB effect ≈ 0.8 % of SSE (D).
- **Unverified:** actual LB effect (labels).
- **Procedure:** D scripts q2b D/D2, q4 L; group by (ADEP, FLIGHT, SCHED_TIME), size-2 groups with MVT within 120 s.
- **Flag:** `no-training`.

#### 7. `false-alarm-non-lirf`
- **Sources:** M `false-alarm-non-lirf`.
- **Hypothesis:** Model B fires 15–20k predictions on large-mms rows at non-LIRF airports whose labels are normal (sched-filled fraction ≈ 0); the ranking set (EHAM) contains many such rows.
- **Verified:** Non-LIRF unmatched, mms in (14,400, 43,200]: 368 rows, sched-filled fraction 0.000; (43,200, ∞): 46 rows, 0.000. Top-25 SSE includes EDDF y=661 pred 16,880 (mms 89,185), EHAM y=549 pred 15,335 (mms 151,639), LFPG y=652 pred 15,812. Ranking EHAM 559 rows mms ≥ 2 h, 148 ≥ 4 h, 8 ≥ 12 h; v5 predicted 7–10k on those, v8 2–7k, v4 0.3–5k. v4 predicted lower than v8/v5 on large-mms non-LIRF rows (EHAM 85,535 → 2,865 vs v5 8,626) (M). Non-LIRF unmatched ms ≥ 2 h in OOF: 1,837 rows, 6.4 % of SSE (D).
- **Unverified:** share of v4's LB advantage attributable to lower non-LIRF predictions; whether an ADEP × mms interaction fixes it inside the GBDT.
- **Procedure:** OOF unmatched rows with ADEP ≠ LIRF by mms band, pred vs y; `outputs/*.parquet` EHAM rows with mms ≥ 2 h compared across submissions.
- **Flag:** `no-training`.

#### 8. `cond-mean-vs-median`
- **Sources:** M `cond-mean-vs-median`.
- **Hypothesis:** For LIRF & mms ≥ 12 h the label is two-valued {mms (62 %), ≈87k (38 %)}; the RMSE-optimal prediction is the mixture mean (~mms + 10k); a median/robust loss would pick mms.
- **Verified:** Bin (43,200, ∞) at LIRF: 29 rows, 62.1 % with y = mms; the rest are 87k rollover rows. Forward fold LIRF & mms ≥ 4 h (n=10): mean y 48,907, model 26,296, bin-mean 44,405, mms 42,960 (M).
- **Unverified [supposed]:** for mms = 60k the mixture mean ≈ 70k gives RMSE ≈ 13k vs ≈ 16.6k for predicting mms and ≈ 25–50k for the current plateau.
- **Procedure:** on `oof_main-48c19bc830` LIRF unmatched rows with mms ≥ 43,200: RMSE of pred = mms, pred = leave-one-month-out bin mean, pred = model.
- **Flag:** `no-training`.

#### 9. `mms-mixture-reparam`
- **Sources:** D `lirf-fill-mixture-prediction`; M `mms-reparam-or-mixture`; E H2 (gate-hold decomposition T = (MVT − SCHED) − G).
- **Hypothesis:** Model B should predict P(sched-filled | x)·mms + (1 − P)·taxi — equivalently, model the target y − mms on LIRF rows, or a gate hold G on top of MVT − SCHED; a two-variable lookup already beats the GBDT on unmatched rows.
- **Verified:**
  - OOF: LIRF unmatched ms ≥ 2 h = 239 rows = 34.8 % of total OOF SSE. Filled subset 129 rows: 5.4 % of SSE (pred median 8,639 vs y 10,560). Non-filled 110 rows: 29.3 % of SSE (pred median 6,091 vs y median 1,112, RMSE 14,172) (D).
  - Separability on 627 LIRF unmatched ms ≥ 2 h training rows (fill 63 %): by ms hours 2 h 60 %, 3 h 54 %, 4 h 73 %, 5 h 87 %, 6 h 89 %, 7–13 h 100 % (n=34); by take-off hour UTC 0–6 h 82–100 %, 10–13 h 43–53 %, 15 h 31 %; by FLIGHT prefix ISR/AEZ/TWB/LAV/MSC/ETH 100 %, NOS 89 %, WZZ 69 %, WMT 60 %, RYR 48 %, EJU 27 %, ITY 21 %, UAL 10 %; FLIGHT_mvt not seen among matched training flights → 92 % (n=89) vs 58 % (n=527). AIRCRAFT_TYPE_mvt null on all 627; FLIGHT_RULE all "I"; runway 25 in 554/627 (D).
  - Leave-one-month-out lookup mean(y | ADEP × mms-bin) vs model B on unmatched OOF rows: 1,755 vs 2,296; fold-honest forward 1,262 vs 2,109; jul 1,458 vs 2,009; jan 2,961 vs 3,223. Replacing only LIRF predictions with the table moves overall OOF 334.58 → 298.05. Mixture with LOMO table P(sched-filled | ADEP, actype-null, mms-bin): 1,937 (soft) / 1,897 (hard at P ≥ 0.5) vs 2,296. Oracle "sched-filled → mms" alone: 2,172 (rollover rows dominate) (M).
  - External: OpenAir Stage4 LIRF-unmatched gate-delay decomposition T = D − G, D = MVT − SCHED, G = gate hold predicted by CatBoost on all LIRF rows with taxi-in / stand-turnaround features; progression 316.97 → 288.90 attributed to it. Phoenix-Ops: LIRF-unmatched specialist CatBoost on 1,488 rows, 5-seed avg (E).
- **Unverified:** no trained version exists (classifier on 22,470 rows, or model B target y − mms on LIRF rows); E H2's pre-check corr(TAXITIME, MVT−SCHED) on LIRF unmatched vs LFPG unmatched not reported; CV → LB transfer.
- **Procedure:** (a) no-training pre-check: corr(TAXITIME, MVT − SCHED) on LIRF unmatched vs LFPG unmatched (E H2). (b) fit: model B with target y − `mvt_minus_sched` on LIRF rows, or P(sched-filled) classifier on the 22,470 unmatched rows using ADEP, actype-null, mms-bin, FLIGHT prefix, take-off hour, seen-among-matched; compare unmatched OOF RMSE against 2,296 on `oof_main-48c19bc830` (M).
- **Flag:** `model-B-fit (~1–2 min)` (M: "yes (cheap)").

#### 10. `tree-identity-saturation`
- **Sources:** M `tree-identity-saturation`.
- **Hypothesis:** Model B saturates at ~35–40k because trees cannot represent y = mms and `min_data_in_leaf` 50 exceeds the number of LIRF training rows with mms > 43,200 (29).
- **Verified:** Ranking LIRF rows mms 47k–111k → v8 predicts 28–40k (one row 68k); ab_split (255/20) 50–71k incl. 71,322 for mms 47,705; OOF sched-filled rows with mms ≥ 2 h have pred/mms median 0.83, p10–p90 [0.55, 1.10]. LIRF training rows with mms > 43,200 = 29 < min_data_in_leaf 50; in (14,400, 43,200] = 98 (M).
- **Unverified:** whether a smaller `min_data_in_leaf` restores pred/mms ≈ 1 on LIRF without harming non-LIRF rows (see item 7).
- **Procedure:** forward fold with `unmatched_min_data_in_leaf 5`; compare pred/mms on LIRF sched-filled OOF rows (M).
- **Flag:** `forward-fold run (~5 min)` (M: "partly").

#### 11. `linear-tree-model-b`
- **Sources:** M `linear-tree-modelB`; P 1 (config resolution shows v4 model B `linear_tree True`); H §4 (v4 = "+ linear_tree", current best).
- **Hypothesis:** v4's 37 s LB advantage over v7/v8 is mostly `linear_tree` in model B extrapolating along mms on ~2 ranking LIRF rows.
- **Verified:** v4's model B used linear_tree (`git show 9f31cef:src/prc/train.py`: model B called `_fit(..., cfg, ...)`; `lgb_params()` adds linear_tree; `for_unmatched()` did not exist) (M); `configs/submitted_v4.yaml` resolves model B to leaves 255, min_data 20, linear_tree True, rounds 3000 on today's code, and 127/50/1500 under v4-era code (P 1). On ranking LIRF rows v4 predicted ≈ mms (108,552 for mms 111,654; 79,686 for 87,484) while v8 saturated at 40,267 / 34,552. RMS(v4 − v8) = 2,967 on mms ≥ 2 h rows vs 469 elsewhere. v4 predicted lower than v8/v5 on large-mms non-LIRF rows (EHAM 85,535 → 2,865 vs v5 8,626) (M). v9 (v8 matched + v4 unmatched) = 306.46; v9 − v4 = 1.43, v9 − v8 = 37.34 (H §4). Code diff 9f31cef → a7a300d for model B: only force_row_wise, for_unmatched() (same 127/50/2000 via config), explicit bagging_seed/feature_fraction_seed — nothing structural (M). Counter-fact from P 4: up_v3 (no linear_tree) already has 16 unmatched predictions > 43,200, so linear_tree is not necessary for the > 12 h predictions.
- **Unverified [supposed]:** if the two labels equal mms (training p = 0.62 for that bin), their SSE difference (≈7.8e9) is ~90 % of the total v4→v8 SSE gap ((343.80² − 305.03²)·344,841 ≈ 8.7e9). Not checked: whether linear_tree on model B over-extrapolates on non-LIRF large-mms rows under current code (M "Not checked").
- **Procedure:** (a) done (file comparison above); (b) forward-fold run with `use_linear_tree true` for model B (~5 min), compare LIRF sched-filled pred/mms; (c) probe submission v4-matched + candidate-unmatched (M).
- **Flag:** `forward-fold run (~5 min)` for step (b); step (c) is `probe submission` (listed in Batch D).

#### 12. `rule-overlays-extreme-rows`
- **Sources:** E H5.
- **Hypothesis:** Rule overlays used by public teams for extreme rows transfer to this pipeline: AOBT_3 − LOBT > 7200 and both proxies in [−7200, 172800] → predict max(0, MVT − LOBT); OPDI ground time ≥ 7200 s → shrink 50 % toward median.
- **Verified (external only, not local):** Phoenix-Ops v11 clock-overlay rule −0.2 on 21–31 rows; OpenAir Stage5 OPDI ground ≥ 7200 → 50 % toward median (~146 rows), 281.72 → 278.36; sergiuv11 LIRF overlay at 14,400 s (E A3).
- **Unverified:** everything locally.
- **Procedure:** apply the rules to training rows / existing OOF predictions and measure RMSE change; no training (E H5).
- **Flag:** `no-training`.

#### 13. `tail-postprocessing-cap-shrink`
- **Sources:** E H6. (Conflicts with item 14; kept as a separate item.)
- **Hypothesis:** Post-processing model B's tail (cap, or shrink toward the airport-hour median) improves RMSE.
- **Verified:** nothing by the source. Counter-evidence in item 14 (clip at 7,200 → 3,373; shrink 0.5·pred + 0.5·airport-median → 2,825, both vs 2,296).
- **Unverified:** cap-quantile sweep on existing OOF; airport-hour (not airport) median; count of ranking predictions > 4 h vs training rate (partly covered by items 3 and 23: submissions predict ≥ 2 h on 103–130 rows vs ≈97 implied; ≥ 12 h on 3–19 vs ≈6; P's n>43200 column 2/16/18/2/2/2).
- **Procedure:** on existing OOF, RMSE vs cap quantile; ranking predictions > 4 h vs training rate (E H6).
- **Flag:** `no-training`.

#### 14. `robust-loss-harmful`
- **Sources:** M `robust-loss-harmful`. (Conflicts with item 13; kept as a separate item.)
- **Hypothesis:** Huber/quantile/log-target/clipping worsen RMSE here because the tail is predictable signal.
- **Verified:** On `oof_main-48c19bc830` unmatched rows: model 2,296; clip 3,600 → 3,568; 7,200 → 3,373; 14,400 → 3,070; 43,200 → 2,306; shrink 0.5·pred + 0.5·airport-median → 2,825; constant airport-median → 3,724; oracle airport-mean → 3,492. Rows y ≥ 4 h: mean y 38,198, mean pred 25,800 (M).
- **Unverified:** loss-function changes (Huber / quantile / log1p target) — not fit.
- **Procedure:** model B fit with `objective` huber / quantile / log1p target; compare unmatched OOF against 2,296 (M: "clip no; loss yes").
- **Flag:** `model-B-fit (~1–2 min)` (the clipping part is already measured, no-training).

### Theme 2 — distribution shift 2025 → 2026

#### 15. `eham-jan2026-disruption`
- **Sources:** D `eham-jan2026-disruption`.
- **Hypothesis:** EHAM Jan 2026 ranking contains a multi-day disruption (Jan 2–7, 9, 29) with 68–123 unmatched departures/day (normal 5–15); training has almost no analogue.
- **Verified:** EHAM unmatched Jan 2026 = 805 vs Jan 2025 = 293 (Jul 401 vs 515). Per day Jan 2026: d2 94, d3 108, d4 123, d5 84, d6 68, d7 18, d9 49, d29 69, else 1–19; Jan 2025 max 37. Proxy sd Jan 2026 d3 1,502, d4 1,963, d5 1,848, d6 1,644, d7 1,440 vs Jan 2025 220–600. AOBT_3 minute-rounding EHAM 0.676 (Jan 2026) vs 0.749 (Jan 2025) — only airport not ~100 % rounded. METAR de-icing-gate hours EHAM 12.8 % vs 5.1 %. Training EHAM unmatched on Jan 5 & 18 2025 (n=72): median 1,006, p90 3,580 vs other Jan days (n=221) 516 / 1,233. `outputs/main.parquet` EHAM Jan unmatched median 1,524, p90 3,152 (D). Ranking EHAM 559 rows mms ≥ 2 h, 148 ≥ 4 h, 8 ≥ 12 h (M).
- **Unverified:** LB effect.
- **Procedure:** D scripts q1 A, q2b C7, q4 K, q5 P.
- **Flag:** `no-training`.

#### 16. `seasonal-deicing-shift`
- **Sources:** D `winter-2026-deicing-shift`; E H11 (Jan/Jul seasonality shift; de-icing).
- **Hypothesis:** Jan 2026 weather (2–4× more de-icing-gate hours than Jan 2025) and Jan/Jul seasonality (elevated mean/SD) place the ranking months outside what the 2025-trained model saw.
- **Verified:** De-icing-gate hour share Jan 2026 / Jan 2025: EHAM 0.128/0.051, LSZH 0.180/0.059, EDDM 0.112/0.069, LTFM 0.095/0.000, EDDF 0.083/0.023, LFPG 0.015/0.031, EGLL 0.001/0.011. Low-visibility lower in 2026 at EHAM (0.031/0.109), LFPG (0.027/0.073). Weather cache complete; null parity ≤ 0.02 pp (D). External: Jan/Jul 2025 mean 995/1026 s, SD 605/745 vs mid-year; LIRF SD 1,332 s (E A3, ahmetabdullahgultekin). H §4/§7: weather added v2→v3 moved LB 312.02 → 306.29, but v3 and v4 were trained with the weather-join bug that overwrote and dropped `hour_utc`.
- **Unverified:** per-month OOF RMSE; residual by airport Jan vs rest; weather effect re-measured with the fixed join (E H11).
- **Procedure:** per-month OOF RMSE on `oof_main-48c19bc830`; residual by airport, Jan vs rest; D script q4 K for the METAR shares.
- **Flag:** `no-training`.

#### 17. `unmatched-composition-shift`
- **Sources:** D `unmatched-composition-shift`; P 8 (share figures).
- **Hypothesis:** Ranking unmatched rows are a different population from training unmatched rows (higher rate, EHAM/LSZH/KLM heavy, fewer "flights seen among matched").
- **Verified:** Unmatched rate 1.53 % (5,283; 2,449 Jan / 2,834 Jul) vs 1.08 % (22,424) [FLIGHT_ID_mvt null; D]. By airport ranking/training: EHAM 3.16/1.65, LSZH 2.85/1.70, LEBL 1.47/1.03, LIRF 1.42/0.93, EDDF 1.16/0.85, LEMD 0.64/0.44. Training rate by month 0.62–2.06 % (peak Jul 2025). Unmatched airport share: EHAM 22.8 vs 18.2, LSZH 12.5 vs 10.2, LFPG 13.9 vs 16.8, LTFM 13.6 vs 16.2. FLIGHT prefix KL 7.4 vs 3.0, TH 10.8 vs 14.5, HB 2.7 vs 4.5. FLIGHT_RULE V 7.9 vs 13.1. Seen among matched flights 69.7 vs 81.9 %. Null pattern identical. Hour distribution near-identical. Training label by seen: median 962 / gt2h 2.08 % vs 846 / 2.27 % (D). P 8 (proxy-null): unmatched share train 1.08 % vs ranking 1.53 % (Jan 2,453, Jul 2,837); training unmatched/month 1,082 (Mar)–3,938 (Jul). H §5: adversarial-validation AUC 0.807 between 2025 training and 2026 ranking; reweighting did not change verdict ordering.
- **Unverified:** effect on model B error.
- **Procedure:** D scripts q1 A/B, q3 G/G2.
- **Flag:** `no-training`.

#### 18. `sched-anchored-delay-shift`
- **Sources:** D `sched-anchored-delay-shift`.
- **Hypothesis:** 2026 delay levels differ by month from 2025, shifting every SCHED/EOBT-anchored feature.
- **Verified:** Matched ms median Jan 2026 vs Jan 2025: EDDF 1,545/1,183, EDDM 1,324/1,025, LTFM 1,615/1,194, LSZH 1,509/1,232, EHAM 1,451/1,371; Jul 2026 vs 2025: EDDF 1,456/1,693, EDDM 1,255/1,374, LIRF 1,864/2,038, LTFM 1,262/1,436, EGLL 1,678/1,918. `aobt3_minus_sched` median Jan 420 vs 240, Jul 481 vs 720. `aobt3_minus_eobt1` LTFM Jan 0 vs −180, Jul −60 vs +120. Proxy medians stable (≤50 s); proxy sd higher Jan 2026 at EHAM (740 vs 476), EDDF, EDDM, LSZH. Congestion counts, headways, taxi-in means: no shift (D).
- **Unverified:** effect on predictions.
- **Procedure:** D scripts q2b C5/C6, q3 F2.
- **Flag:** `no-training`.

#### 19. `timestamp-precision-clusters`
- **Sources:** E H10.
- **Hypothesis:** MVT_TIME second==0 share differs by airport (different data-source precision) and residual SD differs by cluster.
- **Verified:** MVT_TIME zero-second share ~8.3 % at EDDM/EGLL/LFPG/LIRF/LTFM vs ~1.7 % elsewhere (external probe; locally confirmed 0.081–0.085 vs 0.015–0.018) (E A3). Related: AOBT_3 minute-rounded 96.6 % (H §2); EHAM 0.676 (Jan 2026) vs 0.749 (Jan 2025) (D); BLOCK seconds-of-minute within 6 s of a minute boundary for 100 % of LIRF rows and 74 % of other unmatched rows (M).
- **Unverified:** OOF residual SD by (airport, second==0).
- **Procedure:** OOF residual SD by (ADEP, MVT_TIME second == 0) (E H10).
- **Flag:** `no-training`.

### Theme 3 — pipeline, evaluation, reproducibility

#### 20. `v4-config-not-reproducible`
- **Sources:** P 1.
- **Hypothesis:** `configs/submitted_v4.yaml` on today's code does not build v4: model B silently gets 255 leaves / min_data 20 / 3000 rounds (TrainConfig defaults), `hour_utc` is present, `force_row_wise` is on.
- **Verified:** Resolved through TrainConfig + for_unmatched():

  ```
  configs/submitted_v4.yaml | model A: leaves 127, min_data 50, linear_tree True, force_row_wise True, rounds 1500, seeds 1
                            | model B: leaves 255, min_data 20, linear_tree True, rounds 3000
  configs/main.yaml         | model A: 127/50, linear_tree None, rounds 2000, seeds 3 | model B: 127/50, rounds 2000
  ```
  In v4-era code (`git show d7073bf:src/prc/train.py`, predict_fold) model B used cfg itself → 127/50/1500. Defaults unmatched_num_leaves=255 / min_data 20 / num_rounds 3000 at train.py:93-95 are exactly the v5 setting (325.22). v4 also trained without hour_utc and without force_row_wise. `git show a96e43a:configs/v4_linear_tree.yaml` had `scheme: fast` (field removed); submitted_v4.yaml has no unmatched_* keys (P).
- **Unverified:** consequence — what today's code actually produces from `submitted_v4.yaml`; whether v4 is reproducible at all without the old code path and the weather-join bug.
- **Procedure:** config resolution (done). No run proposed by the source.
- **Flag:** `no-training`.

#### 21. `refit-seed-semantics-changed`
- **Sources:** P 2; M `seed-avg-no-effect-B` (its supposed part: "the v2→v8 lottery came from random-stream changes on a saturating model").
- **Hypothesis:** Commit d182fc1 made the refit pass explicit `bagging_seed`/`feature_fraction_seed`, which changes the booster even at n_seeds 1 while the early-stopping fit is unchanged; v2→v8 (LB 312.02 → 343.80) kept identical model-B rounds (658) but produced a different model.
- **Verified:** `git log -S'bagging_seed' -- src/prc/train.py` → introduced in d182fc1 (2026-09-14); its comment "One seed is the default, and then this is the single fit it was before" is contradicted by a toy check (2,000 synthetic rows, num_leaves 15, 50 rounds, deterministic+force_row_wise): {seed:42} vs {seed:42, bagging_seed:42, feature_fraction_seed:42} → old-vs-old identical True; old-vs-new identical False, RMS diff 0.199. v2 (`git show 4e807b5^:experiments/pipeline.jsonl`) rounds_unmatched 658; det_a/v8 rounds_unmatched 658. Same rounds, yet on unmatched rows RMS(v2 − v8) = 314.5 s (84/5,290 rows identical) vs 38.4 s on matched rows. det_a vs det_b (identical configs, same code): model B rounds bit-identical (commit 8dc2152); model A rounds 1126/1146/1131 vs 1273/1034/1173 (P). det_a = det_b = repro_v2 bit-identical (M). Code diff 9f31cef → a7a300d for model B: only force_row_wise, for_unmatched(), explicit bagging_seed/feature_fraction_seed per seed (M).
- **Unverified:** magnitude on the 5,290 ranking unmatched rows.
- **Procedure:** refit model B (658 rounds, ~1 min) with params only vs with explicit sub-seeds; compare predictions on the 5,290 ranking unmatched rows (P).
- **Flag:** `model-B-fit (~1–2 min)`.

#### 22. `hour-utc-absent-in-v3v4`
- **Sources:** P 4; M `v5-verdict-confounded` (caveat on hour_utc).
- **Hypothesis:** v3/v4's model B was trained without `hour_utc` (weather-join bug, H §7-4) and that absence, not weather or linear_tree, produced the > 12 h LIRF predictions.
- **Verified:** up_v3 already has 16 unmatched predictions > 43,200 with no linear_tree; v7 (weather on, hour_utc on, seeds ×3) has 2. RMS(up_v3 − up_v4) on unmatched = 281 s. Confounded with item 21 (P). v5 also trained before the hour_utc fix (34808da), as was v4; v2 had no weather hence hour_utc present (M); v5 emits 2 predictions > 43,200 on 5,290 rows (P 3).
- **Unverified:** the hypothesis itself.
- **Procedure:** fit model B with `exclude_features [hour_utc]`, weather on, 127/50, one seed, old seed semantics; count unmatched ranking predictions > 43,200 (≥ ~15 confirms; 2 refutes) (P).
- **Flag:** `model-B-fit (~1–2 min)`.

#### 23. `holdout-month-decides-rounds`
- **Sources:** P 5 `holdout-six-rows-decide-rounds`; M `early-stop-holdout-month`; H §9 (fixing model B's round count listed as not done).
- **Hypothesis:** Model B's round count comes from L2 early stopping on one fixed month (the latest training month) containing only a handful of extreme rows, so the rounds — and the tail behaviour — are decided by those rows; the forward fold (October holdout, 44–51 rounds) therefore evaluates a model B unlike production's (December holdout, 199–658 rounds).
- **Verified:** Unmatched rows 22,470; y ≥ 7,200: 481; ≥ 14,400: 119; ≥ 43,200: 33; ≥ 80,000: 15; y ≤ 0: 251. Dec-2025 unmatched (inner holdout, `_inner_holdout_mask` = latest month) = 1,649 rows, 6 with y ≥ 14,400. Same month every run (train.py:235-249); variation across configs comes from the model (P). Forward fold model B stops at 44–51 rounds in every run; jan 175–538, jul 321–479, production 199–658. Forward's inner holdout is October = fewest extremes (4 rows ≥ 4 h, max 25,558); production's holdout is December (6 rows ≥ 4 h, max 87,605) (M). H §6: rounds_unmatched observed 199 / 212 / 585 / 658; holdout ≈1,650 rows with 0–5 extreme rows (see §4).
- **Unverified [supposed]:** at lr 0.05 a 45-round model cannot reach leaf means of 40k+ from base ~1,400; screening on `folds: [forward]` evaluates a model B unlike production's.
- **Procedure:** forward-fold run with fixed unmatched rounds (e.g. 600) or `inner_holdout_months 2`; compare unmatched OOF and pred/mms on LIRF rows (M).
- **Flag:** `forward-fold run (~5 min)`.

#### 24. `holdout-encoding-self-inclusion`
- **Sources:** P 6.
- **Hypothesis:** Label encodings (`ref_*`, `arh_med`, `apt_med`) are fitted on the whole training side including the inner-holdout month, so the early-stopping holdout sees its own labels.
- **Verified:** predict_fold (train.py:317-318) fits ref/arh/apt/op on all labelled training rows, then `_fit` carves the holdout month out of the same frame; holdout rows' ref_p10/arh_med/apt_med contain their own labels (diluted by reference_min_count=30, arh min 10). Guard G2 tests only the evaluation fold's labels, not the holdout month's via encodings (P). H §5: removing the 9 encoding columns costs +37.37 all rows / +1.25 (drop 200) / +1.24 (drop 2,000) / +450 unmatched.
- **Unverified:** effect on best_iteration and on the tail.
- **Procedure:** fit encodings on months 1–11, attach to Dec, same early-stop; compare best_iteration (P: "B cheap; A expensive").
- **Flag:** `model-B-fit (~1–2 min)`.

#### 25. `n-features-72-mislabel`
- **Sources:** P 7.
- **Hypothesis:** "72 features" (H §2, "unchanged since day one") is a frame width including `month`, not the matrix width, which changed between builds.
- **Verified:** Old numeric_feature_columns (`git show d7073bf:src/prc/features.py:293-300`) excluded TAXITIME_SEC_mvt but included month; 72 = 71 frame numerics + month. Current cache: numeric_feature_columns 99; visible_numeric_columns(main) 89 on raw frame, 98 after attach_encodings, +10 categoricals ⇒ 108 matrix columns. Weather adds 18. Derived: v2/v8 model A 71+9=80 numeric; v3/v4 70(−hour_utc)+18+9=97; today 98. n_features semantics in experiments.jsonl changed at 97e67c6. window_coverage_60m is cast to Int32 (features.py:227-229) and holds only {0,1} — constant-ish, consistent train/rank (P).
- **Unverified:** nothing further about the count (bookkeeping item; see §4).
- **Procedure:** none further; `visible_numeric_columns()` in src/prc/train.py is the single source of truth (H §10).
- **Flag:** `no-training`.

#### 26. `fold-shape-vs-ranking`
- **Sources:** P 8.
- **Hypothesis:** No CV fold has the production shape (12 months → Jan+Jul 2026); `overall` silently discards `jul_fwd`; unmatched share differs between train and ranking.
- **Verified:** Ranking months 2026-01 (152,719 rows), 2026-07 (192,122). Folds (cv.py:96-121): jan, jul, jul_fwd (Jan–Jun → Jul), forward (Jan–Oct → Nov–Dec). run_cv builds OOF with pl.coalesce in fold order, so jul_fwd predictions never enter `overall`: overall.n = 672,428 while Σ fold n = 863,141. Unmatched share train 1.08 % vs ranking 1.53 %. Routing rule `mvt_minus_aobt3` null in both paths (train.py:329,348): ranking 5,290 such rows, 7 with FLIGHT_ID_mvt (train 46 of 22,470) — consistent (P). H §5: only `forward` predicts the future from the past.
- **Unverified:** consequence for CV/LB agreement.
- **Procedure:** none proposed by the source (structural facts in cv.py).
- **Flag:** `no-training`.

#### 27. `decision-rule-blind-to-tail`
- **Sources:** P 9; E H12.
- **Hypothesis:** `prc.compare` adopts only on the drop-200 CI, ignores the month bootstrap, and by construction removes the ~20 rows that moved the LB; screening resolution (≈0.25 s) cannot see structural wins, which in public logs are 2–28 s while single features give ≤1.4 s.
- **Verified:** verdict() (compare.py:518-546) uses only robustness_curve; month_bootstrap_rmse_diff printed (line 584) but unused; robustness_curve (line 470) drops the K rows with the largest max(se_a, se_b) — exactly where two builds disagree most (the ~20 LIRF rows). paired_bootstrap_rmse_diff (cv.py:157-187) arithmetically correct. A `folds: [forward]` OOF compared with a 3-fold OOF silently restricts to Nov–Dec rows (P). H §5: unmatched-only paired 95 % CI width 62 s vs 0.77 s matched; the ten feature experiments scored −0.21…+0.09 vs noise control; the rule agreed with LB direction in two known cases (weather, model B 255 leaves). External step sizes: OpenAir 316.97 → 288.90 → 281.72 → 278.36 → 276.99; Phoenix-Ops v7 −2.6, v8 −7.1, v9 −0.9, v10 −2.6, v11 −0.2, v12 −1.4; youthful-giraffe v13 one-step −14 s (E).
- **Unverified:** which past adopt/reject verdicts would flip under a different rule.
- **Procedure:** code reading done; re-run `make compare` on existing OOF pairs with the month bootstrap and without the drop-200 restriction and list flipped verdicts.
- **Flag:** `no-training`.

#### 28. `cache-fingerprint-gaps`
- **Sources:** P 10.
- **Hypothesis:** The feature-cache fingerprint hashes only `features.py`; changes to `weather.py`, `airports.py`, the METAR cache or `train.py` do not invalidate it.
- **Verified:** cache_fingerprint (pipeline.py:250-266) = raw file names/sizes, row counts, use_weather, sha256(features.py) only. Current cache feature_code 590ceb8f8170ff6b == sha256(features.py) now; built 09-15 11:38:20 after last features.py commit (11:37:52); weather.py last changed 34808da (07:49) — consistent today. Not in key: weather.py, airports.py, data/ref/metar_hourly.parquet, METAR CSVs, train.py (P).
- **Unverified:** nothing today (current cache consistent); a risk for future runs only.
- **Procedure:** none further.
- **Flag:** `no-training`.

#### 29. `v5-verdict-confounded`
- **Sources:** M `v5-verdict-confounded`; P 1 (defaults = v5 setting).
- **Hypothesis:** The "255/20 cost 13 s" verdict (H §6: 312.02 → 325.22) compared v5 (new code) with v2 (old code); against same-code v7/v8, v5 is 17–19 s better.
- **Verified:** v5 325.22 vs v7 342.59 / v8 343.80 (same code family). Implied unmatched RMSE v5 1,961 vs v8 2,157. v5 trained before the hour_utc fix (34808da), as was v4; v2 had no weather hence hour_utc present. v5 vs v8 on the 11 ranking LIRF mms ≥ 12 h rows: v5 30–61k, v8 29–69k — nearly the same (M). ab_split (255/20) OOF predicts 50–71k incl. 71,322 for mms 47,705 (M, item 10). TrainConfig defaults 255/20/3000 are exactly the v5 setting (P 1). H §5: the CV rule agreed with LB direction for "model B 255 leaves".
- **Unverified [supposed]:** the v5 gain over v8 is elsewhere than the 11 LIRF ≥ 12 h rows; 255/20 vs 127/50 under identical current code.
- **Procedure:** probe submission (M: "probe").
- **Flag:** `probe submission`.

### Theme 4 — model structure and features

#### 30. `shared-b-all-rows`
- **Sources:** E H7; H §8 (implemented, then removed, never measured).
- **Hypothesis:** Training model B on all ~2.08 M rows (shared_b, schedule-only features) improves unmatched predictions.
- **Verified:** nothing measured (H §8). Pre-check proposed: per-airport y distribution unmatched vs matched (E).
- **Unverified:** the run.
- **Procedure:** pre-check (no training), then one training run with the shared_b configuration (E: "One training run"). The same run settles item 31.
- **Flag:** `full run (~20 min)`.

#### 31. `shared-b-no-transfer`
- **Sources:** M `shared-b-no-transfer`. (Conflicts with item 30; kept separate.)
- **Hypothesis:** Training model B on all rows will not help the tail because the LIRF actype-null sched-filled regime does not exist among matched rows.
- **Verified:** Matched rows with |BLOCK−SCHED| ≤ 120 s are 11–26 % per airport but matched y ≥ 2 h is 103 rows of 2.06M (47 in OOF, 0.9 % of matched SSE, y − proxy median 6,960); of 122 training rows y ≥ 4 h, 119 unmatched; actype-null among matched LIRF: 0 (M). LIRF matched fill 17.8 % (D, item 4).
- **Unverified:** the run itself (M's table: needs training "yes" — it is the same run as item 30).
- **Procedure:** E H7's pre-check (per-airport y distribution unmatched vs matched) is no-training; the training run is shared with item 30.
- **Flag:** `no-training` here for the pre-check; the source's own flag is "yes" (training), covered by item 30's run.

#### 32. `arrival-turnaround-features`
- **Sources:** E H3.
- **Hypothesis:** Arrival/turnaround features from ARR rows (taxi-in, in-block time, previous arrival on the same stand) carry signal, especially for unmatched rows.
- **Verified (external):** Phoenix-Ops v7 arrival features (26 taxi-in/same-stand features from ARR rows) −2.6; v8 deeper arrival residual −7.1; OpenAir taxi-in features 288.90 → 281.72 (E A3). Local: arrivals in ranking carry full TAXITIME/BLOCK; taxi-in medians within ~50 s except EHAM Jul (335 vs 412) (D `checked-not-a-cause`); taxi-in means show no 2025→2026 shift (D, item 18).
- **Unverified:** coverage and correlation locally; CV effect.
- **Procedure:** join DEP → latest ARR same airport + stand; coverage and corr with y by matched/unmatched (no-training pre-check); training run to score (E).
- **Flag:** `full run (~20 min)`.

#### 33. `peer-following-context-features`
- **Sources:** E H4.
- **Hypothesis:** Peer "following-context" NM clock features (mean peer MVT−AOBT_3, AOBT_3−EOBT_1, ±30 min, leave-one-out, including windows after the query time) explain model-A residual.
- **Verified (external):** Phoenix-Ops v9 NM-neighbour peer MVT−AOBT_3 proxies (30 features) −0.9; v12 "following-context" NM features (windows strictly AFTER query time, 15/60 min) −1.4; OpenAir Stage2 "disruption state" (mean peer lateness 30-min, self-excluded) (E A3). Local: H §5 congestion-type features (`queue_ahead` −0.16 [−0.40,+0.08], `bank_position` −0.19 [−0.41,+0.07], `active_runways_60m` −0.06, `dep_arr_ratio_30m` +0.03, `congestion_vs_usual_30m` +0.09 [−0.16,+0.31]) all rejected vs noise control on `folds: [forward]`; OpenAir report: rolling queue features did not shrink their tail (E).
- **Unverified:** corr with model-A residual locally; CV effect.
- **Procedure:** corr with model-A residual (pre-check); training run (E).
- **Flag:** `full run (~20 min)`.

#### 34. `airport-specialists-blend`
- **Sources:** E H8.
- **Hypothesis:** Per-airport specialist models and a CatBoost/LGBM NNLS blend beat the single global LightGBM.
- **Verified (external):** OpenAir Stage3 NNLS blend CatBoost 45.6 % + airport LGB experts 49.1 % + XGB 5.3 %; Phoenix-Ops v10 CatBoost −2.6; sergiuv11 single global LightGBM (127 leaves, min_leaf 200, 436 trees) + airport specialists blended 50/50 at LB 286.66; NASA pushback winners all per-airport models; 2025 2nd jubilant-vase CatBoost 64 %/LGBM 20 %/XGB 16 % (E A3–A5). Local: within-group RMSE on matched normal-range rows (H §5): airport 368.2; airport × runway 354.6; × stand 307.5; × hour 284.9; × aircraft type 268.1; current model ≈202.
- **Unverified:** OOF residual by airport × month locally; CV effect.
- **Procedure:** pre-check OOF residual by airport × month; training runs (E).
- **Flag:** `full run (~20 min)`.

#### 35. `explicit-clock-difference-features`
- **Sources:** E H9.
- **Hypothesis:** Explicit clock differences (AOBT_3−EOBT_1, LOBT−EOBT_1, IOBT−EOBT_1) and a P10 unimpeded reference per (airport, stand, runway) add signal.
- **Verified:** D reports medians of columns named `aobt3_minus_sched` and `aobt3_minus_eobt1` (item 18), so at least those exist in the frame; encodings include `ref_p10` (P 6). External: ahmetabdullahgultekin P10 unimpeded reference per (airport, stand, runway) per ATXOT; OpenAir Stage1 per-airport OLS on MVT−AOBT and AOBT−EOBT (E A3); ATXOT unimpeded = median per runway × stand group (≥10 flights) at sub-saturation (E A5).
- **Unverified:** whether LOBT−EOBT_1 and IOBT−EOBT_1 are in `features.py` and visible to the models; corr with residual; CV effect.
- **Procedure:** grep `src/prc/features.py`; corr with residual (no-training); training run (E).
- **Flag:** `full run (~20 min)`.

---

## 2. Verified negatives (do not re-investigate)

| id | source | what was checked | result |
|---|---|---|---|
| `alignment-weather-probe-checked-ok` | P 11 | Row alignment, Int32 rounding, weather coverage for 2026, month-edge rows, v9 probe construction | All seven submission files: 344,841 rows, ids in submitting.parquet order, MVT_ID_mvt Float64 / TAXITIME_SEC_mvt Int32 (np.rint, ≤0.5 s). v9 probe exact: RMS(v8 − v9) matched = 0.0; RMS(up_v4 − v9) unmatched = 0.0. METAR covers 2025-01-01 → 2026-07-31 all 10 stations; Jan/Jul 2026 744 h/station (LSZH 743); weather null rate feat_rank 0.0 (wind_ms 0.02 % Jul), feat_train ≤ 0.12 %. Both frames 132 columns, identical names/order/dtypes; MVT_TIME_UTC_mvt UTC-aware. Month-edge context rows 56. Unseen categories in ranking: STAND 284 rows, OPERATOR 177, ADES 174, AIRCRAFT_TYPE 18, RUNWAY 0. Side fact: backfilled uploaded_at for v1–v4 are ~9 h later than local outputs mtimes (JST labelled +00:00); affects only the daily counter. |
| `library-drift-refuted` | P 12 | Library version change 09-10 → 09-15 as cause of v4 → v8 | lightgbm 4.7.0, polars 1.44.2, numpy 2.4.6, pyarrow 25.0.1 dist-info all dated 2026-09-10 07:25 (before v1–v4). No lockfile; pyproject pins nothing. |
| `checked-not-a-cause` | D | Label identity, negative labels, weather coverage/parity, null parity, ID overlap, timezone types, routing of AOBT-null matched rows | TAXITIME == MVT − BLOCK within 1 s on 100 %; no null labels. Negative labels 369 rows (min −12 s; LSZH 298, LFPG 61). Weather complete, no nulls in temp/vis/wind (gust 97 % null both sets). Null parity: only NM-derived columns differ; dest_distance_km 0.52 vs 0.29 %. Arrivals in ranking carry full TAXITIME/BLOCK; taxi-in medians within ~50 s except EHAM Jul (335 vs 412). All timestamps Datetime(us, UTC). Routing key proxy-null sends 46 training / 7 ranking matched rows to model B. Ranking months Jan 2026 152,719 DEP, Jul 2026 192,122, full coverage. |
| `seed-avg-no-effect-b` | M (verified part) | Seed averaging on model B | Unmatched OOF 2,312.5 (3 seeds, main-ab56551b5e) vs 2,299.0 (1 seed, det_a); LIRF 7,253 vs 7,206. det_a = det_b = repro_v2 bit-identical. (H §9 lists "increase model B seeds" as not done; this measurement covers 3 vs 1.) |
| `clip-shrink-worsens-unmatched` | M `robust-loss-harmful` (measured part) | Clipping / shrinking model B predictions on existing OOF | model 2,296; clip 3,600 → 3,568; 7,200 → 3,373; 14,400 → 3,070; 43,200 → 2,306; shrink 0.5·pred + 0.5·airport-median → 2,825; constant airport-median → 3,724; oracle airport-mean → 3,492. |
| `routing-consistency` | P 8, D | Proxy-null routing train vs ranking | Ranking 5,290 proxy-null rows, 7 with FLIGHT_ID_mvt; train 46 of 22,470 — consistent. |
| `handoff-rejected-features` | H §5 | Ten single-feature screens on `folds: [forward]` vs noise control, drop-200 | `queue_ahead` −0.16 [−0.40,+0.08]; `bank_position` −0.19 [−0.41,+0.07]; `stand_rwy_share` −0.10 [−0.38,+0.14]; `active_runways_60m` −0.06 [−0.29,+0.17]; `dep_arr_ratio_30m` +0.03 [−0.16,+0.22]; `congestion_vs_usual_30m` +0.09 [−0.16,+0.31]; `op_offset_med`/`op_offset_n` rejected; `dest_distance_km` rejected (tail only); `planned_flight_s` −0.21 [−0.49,+0.08] (matched 218.5 → 218.1). All rejected. (Item 27 questions the rule, not the numbers.) |
| `handoff-ablations` | H §5 | A/B split vs one model; hyperparameters; encodings; native categoricals; adversarial reweighting | Split: all rows −74.10 [−103.51,−44.44]; drop 200 −6.26 [−6.97,−5.55]; drop 2,000 −6.04 [−6.37,−5.73]; matched 226.2 → 218.5, unmatched 2,697.2 → 1,875.3. Six hyperparameters to LightGBM defaults: −0.94 (CI crosses 0) / +1.41 / +1.58 / unmatched −24. Remove 9 encodings: +37.37 / +1.25 / +1.24 / +450. Remove native categoricals: +7.38 / +7.16 / +4.68 / +33. Adversarial AUC 0.807; reweighting did not change verdict order. |

Reported by other teams, **not verified locally** (E A3, OpenAir): rolling queue features did not shrink the tail; previous-departure block intervals corr≈0.008; unmatched LFPG "nearly random".

Noted by a source as explicitly **not checked** (M "Not checked"): whether linear_tree on model B over-extrapolates on non-LIRF large-mms rows under current code (folded into item 11); whether an `mvt_minus_sched`-monotone constraint helps ([supposed] wrong for non-LIRF); label plausibility of 251 unmatched / 137 matched rows with y ≤ 0 (negligible SSE).

---

## 3. Execution batches

Machine: 4 cores, 15 GB RAM. `feat_train.parquet` is 2,085,047 rows × 132 columns; D's q-scripts took 1–3 min each at `POLARS_MAX_THREADS=2`.

### Batch A — no-training (21 items; may run in parallel, at most ~6 concurrent polars jobs with `POLARS_MAX_THREADS=1`)

| # | id | main inputs |
|---|---|---|
| 1 | `sched-filled-block` | feat_train, feat_rank, oof_main-48c19bc830 |
| 2 | `block-day-rollover` | feat_train, feat_rank, oof_main-48c19bc830 |
| 3 | `lirf-regime-tail-count` | outputs/*.parquet, submitting.parquet, oof_main-48c19bc830, probe_unmatched_from_v4.parquet |
| 4 | `lirf-matched-fill` | feat_train, oof_main-48c19bc830 |
| 5 | `matched-proxy-outliers` | feat_train, feat_rank, oof_main-48c19bc830 |
| 6 | `return-to-stand-twins` | feat_train, feat_rank |
| 7 | `false-alarm-non-lirf` | oof_main-48c19bc830, outputs/*.parquet |
| 8 | `cond-mean-vs-median` | oof_main-48c19bc830 |
| 12 | `rule-overlays-extreme-rows` | feat_train, oof_main-48c19bc830 |
| 13 | `tail-postprocessing-cap-shrink` | oof_main-48c19bc830, outputs/*.parquet |
| 15 | `eham-jan2026-disruption` | feat_train, feat_rank, metar_hourly, outputs/main.parquet |
| 16 | `seasonal-deicing-shift` | oof_main-48c19bc830, metar_hourly |
| 17 | `unmatched-composition-shift` | feat_train, feat_rank |
| 18 | `sched-anchored-delay-shift` | feat_train, feat_rank |
| 19 | `timestamp-precision-clusters` | oof_main-48c19bc830 (MVT_TIME) |
| 20 | `v4-config-not-reproducible` | configs/, git history (no data) |
| 25 | `n-features-72-mislabel` | src/prc, experiments.jsonl (no data) |
| 26 | `fold-shape-vs-ranking` | src/prc/cv.py (no data) |
| 27 | `decision-rule-blind-to-tail` | src/prc/compare.py, existing OOF pairs |
| 28 | `cache-fingerprint-gaps` | src/prc/pipeline.py (no data) |
| 31 | `shared-b-no-transfer` (pre-check) | feat_train |

Items 20, 25, 26, 28 are code/config reads and do not count against the polars concurrency limit. Items 9 (a), 32, 33, 34, 35 also have no-training pre-checks that may be run in this batch ahead of their training runs; they are counted in Batches B/C by their decisive step.

### Batch B — model-B-only fits (5 items; one at a time, ~1–2 min each)

| # | id | fit |
|---|---|---|
| 9 | `mms-mixture-reparam` | model B target y − mvt_minus_sched on LIRF rows, or P(sched-filled) classifier on 22,470 rows |
| 14 | `robust-loss-harmful` | model B with huber / quantile / log1p objective |
| 21 | `refit-seed-semantics-changed` | model B, 658 rounds, params-only vs explicit sub-seeds; compare on 5,290 ranking unmatched rows |
| 22 | `hour-utc-absent-in-v3v4` | model B, exclude_features [hour_utc], weather on, 127/50, one seed, old seed semantics; count predictions > 43,200 |
| 24 | `holdout-encoding-self-inclusion` | encodings fit on months 1–11, attached to Dec, same early stop; compare best_iteration |

### Batch C — forward-fold or full runs (8 items; one at a time)

| # | id | run | est. |
|---|---|---|---|
| 10 | `tree-identity-saturation` | forward fold, `unmatched_min_data_in_leaf 5` | forward-fold run (~5 min) |
| 11 | `linear-tree-model-b` (step b) | forward fold, `use_linear_tree true` for model B | forward-fold run (~5 min) |
| 23 | `holdout-month-decides-rounds` | forward fold, fixed unmatched rounds (e.g. 600) or `inner_holdout_months 2` | forward-fold run (~5 min) |
| 30 | `shared-b-all-rows` (also settles 31) | shared_b configuration | full run (~20 min) |
| 32 | `arrival-turnaround-features` | DEP → ARR same airport + stand features | full run (~20 min) |
| 33 | `peer-following-context-features` | peer NM clock features incl. after-query windows | full run (~20 min) |
| 34 | `airport-specialists-blend` | per-airport specialists + blend | full run (~20 min) |
| 35 | `explicit-clock-difference-features` | AOBT_3−EOBT_1, LOBT−EOBT_1, IOBT−EOBT_1, P10 reference | full run (~20 min) |

### Batch D — requires a probe submission (1 item + 1 step; NOT scheduled — needs the user's explicit authorization)

| # | id | probe |
|---|---|---|
| 29 | `v5-verdict-confounded` | same-code 255/20 vs 127/50 comparison via submission |
| 11 (c) | `linear-tree-model-b` step (c) | v4-matched + candidate-unmatched |

Constraints from E A1: 5 submissions/day, 1 GB; ranking.html: "We will monitor submissions for attempts to learn from or exploit the ranking process. We consider such practices unfair and inconsistent with the purposes and goals of the Data Challenge."

---

## 4. Factual discrepancies between the reports (use one definition)

1. **"33 extreme rows" — 12 h, not 4 h.** H §6 says 極端値（y ≥ 4h 相当）は 33 行. P 5 measured on the 22,470 unmatched training rows: y ≥ 43,200 s (12 h) = **33**; y ≥ 14,400 s (4 h) = **119** (≥ 7,200: 481; ≥ 80,000: 15). M confirms 119 unmatched rows y ≥ 4 h (115 LIRF) and 122 training rows y ≥ 4 h in total (119 unmatched). D's buckets (4–20 h: 106; > 20 h: 16 → 122) agree with M's 122. H §3's "y ≥ 4h: 55 rows" is an OOF count (672,428 rows), not a training count.

2. **Unmatched row counts — two definitions.** 22,470 / 5,290 (H, P, M) = proxy null (`mvt_minus_aobt3` / `AOBT_3` null), which is the model-B routing rule. 22,424 / 5,283 (D) = `FLIGHT_ID_mvt` null. Difference: 46 training / 7 ranking matched rows with null AOBT_3, routed to model B. Per-month ranking counts differ accordingly: Jan 2,453 / Jul 2,837 (P, proxy-null) vs Jan 2,449 / Jul 2,834 (D, FLIGHT_ID-null). Training rate 1.08 % and ranking rate 1.53 % are quoted identically under both definitions.

3. **Inner-holdout extreme count.** H §6: holdout ≈1,650 rows containing 0–5 extreme rows. P 5: Dec-2025 holdout = 1,649 rows, **6** with y ≥ 14,400. M: December holdout 6 rows ≥ 4 h (max 87,605); October (forward fold) 4 rows ≥ 4 h (max 25,558).

4. **Rollover rows — different subsets and denominators.** D: 16 training rows y > 20 h (13 with y − 86,400 = 602…1,992); 10 such rows in OOF = 34.7 % of total OOF SSE. M: 8 OOF rows with y ≈ 86,400 + taxi = 52.7 % of **unmatched** SSE (≈25 % of total). H §3: 10 rows y ≥ ~1 day = 24.1 % of total OOF SSE. D counts by y > 20 h (matched + unmatched); M counts unmatched rollover rows only; H's OOF may differ from oof_main-48c19bc830.

5. **Predictions > 12 h per submission.** H §6 / main.yaml header: "v5 predicted over 12 h twice as often as training". P 3 measured on 5,290 unmatched rows: v5 emits **2** predictions > 43,200 (training rate 33/22,470 ⇒ ~7.8 expected); v2 2, up_v3 16, up_v4 18, v7 2, v8 2. D `tail-count-per-set`: submissions predict ≥ 12 h on **3–19** rows (training frequency implies ≈6) and ≥ 2 h on 103–130 (implies ≈97). P's 2–18 range vs D's 3–19 range presumably reflect different row sets (unmatched-only vs all rows) and/or > vs ≥.

6. **Sched-filled fraction at LIRF, ms ≥ 2 h.** D: 64 % on 622 rows (2–20 h band), 63 % on 627 rows (ms ≥ 2 h); M: 0.633 (mms ≥ 2 h). The 622 vs 627 difference is the > 20 h band. Fill thresholds also differ (D ≤ 10 s; M ±6 s / ≤ 120 s / |y − mms| < 1,800 s) — see §0.

7. **RMS(v4 − v8) on unmatched ranking rows.** P: 1735.5 s; M: 1,736 — same quantity, different rounding.

8. **"72 features".** H §2: 72 columns, unchanged since day one. P 7: 72 = 71 frame numerics + `month`; actual matrix widths 80 numeric + 10 categorical (v1/v2/v8), 97 + 10 (v3/v4), 98 + 10 today.

9. **OOF baselines differ across sources.** H: model A 217.7 / model B 1,864.9 (and split ablation 218.5 / 1,875.3). D and M: `oof_main-48c19bc830` total 334.58, unmatched 2,296 (M), LIRF unmatched 537 rows = 36.8 % SSE (D). Do not compare an H figure with a D/M figure as if from the same OOF.

10. **Leaderboard snapshot.** H (2026-09-15): 53 / 127 teams; leader youthful-giraffe 245.02; 2nd enthusiastic-daisy 249.20. E (2026-09-16, data.json): 154 teams; vigorous-whistle 243.58 (13 subs), enthusiastic-daisy 244.83 (49), youthful-giraffe 245.02 (21, last 09-10). The task brief uses 245.02 as "leader".

11. **v4 model B hyperparameters.** H §4/§6 describe v4 as 127 leaves + linear_tree; P 1 shows `configs/submitted_v4.yaml` resolves to 255/20/3000 on **today's** code and to 127/50/1500 on v4-era code (d7073bf); M (9f31cef) confirms v4's model B used cfg's linear_tree. All three agree v4 was 127/50 with linear_tree at the time; the discrepancy is only in what the YAML now means.

12. **Notation.** D writes `ms`, M writes `mms`, the feature is `mvt_minus_sched`; D writes "fill", M writes "sched-filled". D's "proxy" and H's "proxy" are the same quantity (MVT_TIME − AOBT_3).

13. **"≥ 12 h" counts by label vs by mms.** P: unmatched rows with **y** ≥ 43,200 = 33; all training rows y ≥ 43,200 = 34; LIRF unmatched y ≥ 14,400 = 115 of 119. M: LIRF unmatched rows with **mms** > 43,200 = 29 (62.1 % with y = mms); non-LIRF unmatched rows with mms in (43,200, ∞) = 46 (sched-filled fraction 0.000); ranking LIRF unmatched mms ≥ 12 h = 11. These are label-based and mms-based counts respectively and are not interchangeable.
