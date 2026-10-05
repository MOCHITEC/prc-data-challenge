# Extreme predictions (> 10,800 s) in v33: measured facts (2026-10-05)

This is a read-only check. Nothing in `src/` or `configs/` was changed. No model was trained or submitted, and no
decision is made here. Scripts are in the session scratchpad and are not committed. Every read was done with
`POLARS_MAX_THREADS=1` and a `total_rss` gate below 11,800 MB. Peak RSS of each script: 0.4–0.65 GB.

Inputs:
- v33 (team best, LB 250.9496): `outputs/v33_blend_l2_pa_opt_adsb_learned.parquet`, which is the base plus the
  learned ADS-B residual.
- Base: `outputs/hybrid_blend_l2_pa_opt.parquet`.
- Both files are checked to have the ids and order of `data/raw/submitting.parquet` (344,841 rows, all DEP).
- Lane and features: `data/work_flseq/feat_rank.parquet`. Unmatched means `mvt_minus_aobt3` is null.
- Training: `data/work_flseq/feat_train.parquet` (2,085,047 DEP rows, 0 null labels).
- OOF of the current base: `logs/retune/scratch/oof_hybrid_blend_l2_pa_opt.parquet`, written by
  `scripts/blend_pa_opt.py oof --scratch logs/retune/scratch` (optuna-retune.md §3). Joined to labels, it has
  672,428 scored rows (months 1, 7, 11 and 12). RMSE is 268.320, which reproduces the 268.32 in optuna-retune.md.
  This OOF does not include the ADS-B residual.

## 1. v33 predictions on the scored rows, by lane

| lane | n | mean | q50 | q90 | q99 | q99.9 | q99.99 | max | > 3,600 | > 7,200 | > 10,800 | > 30,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched | 339,551 | 1,024.9 | 946 | 1,502 | 2,563 | 4,227 | 6,579 | 84,883 | 721 | 23 | 2 | 1 |
| unmatched | 5,290 | 1,626.1 | 1,053 | 2,522 | 9,480 | 55,897 | 89,051 | 113,641 | 236 | 94 | 45 | 21 |
| all | 344,841 | 1,034.1 | 947 | 1,510 | 2,639 | 4,769 | 15,711 | 113,641 | 957 | 117 | 47 | 22 |

For comparison, the base without the ADS-B residual has 918 / 123 / 50 / 23 rows above the same thresholds, with
the same max of 113,641. The gate warning ("max prediction 113641 exceeds 10800s") comes from one row,
213145579, and that row is identical in the base and in v33.

**All 47 rows above 10,800 s:**
- 46 are LIRF and 1 is LFPG.
- 45 are unmatched. All 45 unmatched rows are LIRF.

### What drives each row

- **The 45 LIRF unmatched rows come from model B's schedule reparameterisation.**
  - Every member config has `unmatched_sched_reparam: true` and `unmatched_sched_reparam_airports: [LIRF]`.
  - So at LIRF, model B fits `y − mvt_minus_sched` and adds `mvt_minus_sched` back to the prediction.
  - The code is `train.py` `sched_reparam_shift` and `unmatched_target`, with `g = g + shift_tg` at train.py:1505.
  - On these 45 rows, the base equals `outputs/main_flseq.parquet` (model B) exactly.
  - `base − mvt_minus_sched` ranges from −6,608 to +1,987, with a median of +133.
  - So the prediction is essentially MVT − SCHED. The rows have no EOBT_1, AOBT_3, LOBT or IOBT (the NM fields are
    all null).
  - The ADS-B residual moved 19 of the 45 rows, by −1,013 to +796 s.
- **The 2 matched rows come from the proxy and the LOBT bound.**
  - The anchored members predict `mvt_minus_aobt3` plus a learned g.
  - The plain-y member (p2o) sits exactly on the LOBT lower bound, `mvt_minus_lobt − 3606`.
  - That bound comes from `lobt_bound: true`, train.py:1526–1561.
  - 213147416 (LIRF): MVT − AOBT_3 = 86,340, which is 24 h minus 60 s. The members are a2o 85,309, pao 86,140,
    p2o 83,214 (= 86,820 − 3,606) and c2 85,711.
  - 212756714 (LFPG): proxy 13,313. The members are a2o 12,398, pao 12,816, p2o 10,014 (= 13,620 − 3,606) and
    c2 12,657.

Column key for the table below:
- `ms` = MVT − SCHED, `me` = MVT − EOBT_1, `ml` = MVT − LOBT, `mA` = MVT − AOBT_3, all in seconds.
- Times are UTC in 2026, in mm-dd hh:mm.
- `Δadsb` = v33 − base.

| MVT_ID | ADEP→ADES | lane | v33 | Δadsb | SCHED | EOBT_1 | AOBT_3 | MVT | ms | me / ml / mA |
|---|---|---|---:|---:|---|---|---|---|---:|---|
| 213145579 | LIRF→LIMC | U | 113,641 | 0 | 07-10 14:50 | – | – | 07-11 21:50 | 111,654 | – |
| 213152353 | LIRF→YSSY | U | 89,051 | 0 | 07-19 10:00 | – | – | 07-20 10:18 | 87,484 | – |
| 213147416 | LIRF→LFMN | M | 84,883 | 0 | 07-16 19:35 | 07-16 21:53 | 07-16 21:52 | 07-17 21:51 | 94,560 | 86,280 / 86,820 / 86,340 |
| 213145102 | LIRF→SBGL | U | 63,058 | 0 | 07-09 20:10 | – | – | 07-10 13:12 | 61,322 | – |
| 213154026 | LIRF→HLLB | U | 62,343 | 0 | 07-16 15:30 | – | – | 07-17 08:20 | 60,656 | – |
| 202680942 | LIRF→LWSK | U | 57,025 | 0 | 01-02 19:45 | – | – | 01-03 11:27 | 56,520 | – |
| 202687258 | LIRF→LMML | U | 55,897 | +419 | 01-22 05:45 | – | – | 01-22 20:37 | 53,520 | – |
| 213153677 | LIRF→SBGR | U | 49,309 | 0 | 07-19 20:00 | – | – | 07-20 09:08 | 47,337 | – |
| 213142469 | LIRF→LBWN | U | 49,007 | 0 | 07-04 17:50 | – | – | 07-05 07:05 | 47,705 | – |
| 213143319 | LIRF→DTTA | U | 48,909 | 0 | 07-05 14:20 | – | – | 07-06 03:52 | 48,776 | – |
| 202685083 | LIRF→LRCL | U | 47,770 | −950 | 01-14 00:40 | – | – | 01-14 14:14 | 48,841 | – |
| 213143669 | LIRF→LICJ | U | 47,349 | −819 | 07-06 08:10 | – | – | 07-06 21:38 | 48,539 | – |
| 202689507 | LIRF→SBGL | U | 41,768 | 0 | 01-24 21:30 | – | – | 01-25 08:45 | 40,504 | – |
| 213142746 | LIRF→LICT | U | 41,511 | −862 | 07-05 10:20 | – | – | 07-05 21:49 | 41,395 | – |
| 202681830 | LIRF→LIBP | U | 40,307 | 0 | 01-05 00:30 | – | – | 01-05 11:46 | 40,615 | – |
| 202680695 | LIRF→LFRS | U | 40,108 | +796 | 01-03 10:15 | – | – | 01-03 21:11 | 39,365 | – |
| 213155642 | LIRF→UDYZ | U | 38,272 | 0 | 07-28 19:10 | – | – | 07-29 05:32 | 37,379 | – |
| 213154831 | LIRF→EGBB | U | 36,616 | −889 | 07-31 10:25 | – | – | 07-31 21:16 | 39,060 | – |
| 213145828 | LIRF→YPPH | U | 35,742 | −1,005 | 07-11 10:00 | – | – | 07-11 20:01 | 36,062 | – |
| 202683962 | LIRF→EPWA | U | 33,363 | 0 | 01-10 18:35 | – | – | 01-11 03:45 | 33,055 | – |
| 202681769 | LIRF→LYBE | U | 32,313 | −573 | 01-04 19:40 | – | – | 01-05 04:33 | 31,983 | – |
| 213147307 | LIRF→LEPA | U | 32,130 | 0 | 07-26 13:00 | – | – | 07-26 22:10 | 33,057 | – |
| 213143126 | LIRF→OMAA | U | 29,765 | −922 | 07-06 05:00 | – | – | 07-06 13:10 | 29,455 | – |
| 202689463 | LIRF→LYBE | U | 28,925 | 0 | 01-24 19:35 | – | – | 01-25 03:08 | 27,183 | – |
| 213142471 | LIRF→LBWN | U | 26,757 | 0 | 07-04 17:50 | – | – | 07-05 00:56 | 25,564 | – |
| 202681707 | LIRF→LYBE | U | 23,354 | 0 | 01-04 17:20 | – | – | 01-05 00:26 | 25,619 | – |
| 202687662 | LIRF→LIMF | U | 23,069 | 0 | 01-22 15:15 | – | – | 01-22 21:25 | 22,256 | – |
| 213154406 | LIRF→HLLB | U | 20,912 | −997 | 07-31 03:00 | – | – | 07-31 08:45 | 20,755 | – |
| 213153920 | LIRF→LIRZ | U | 18,889 | 0 | 07-26 09:30 | – | – | 07-26 14:51 | 19,263 | – |
| 213141089 | LIRF→HLLB | U | 17,605 | 0 | 06-30 22:01 | – | – | 07-01 03:01 | 18,059 | – |
| 213147953 | LIRF→LIEO | U | 17,486 | −897 | 07-20 12:45 | – | – | 07-20 17:42 | 17,822 | – |
| 213150756 | LIRF→LIRN | U | 16,966 | −946 | 07-24 15:40 | – | – | 07-24 20:48 | 18,538 | – |
| 213141005 | LIRF→LEAL | U | 16,098 | 0 | 07-01 19:55 | – | – | 07-02 01:20 | 19,500 | – |
| 202683690 | LIRF→LFPG | U | 16,014 | 0 | 01-09 21:45 | – | – | 01-10 01:55 | 15,058 | – |
| 213151252 | LIRF→LLBG | U | 15,711 | 0 | 07-24 21:55 | – | – | 07-25 02:10 | 15,358 | – |
| 202685556 | LIRF→LIME | U | 15,703 | +660 | 01-15 13:30 | – | – | 01-15 17:59 | 16,198 | – |
| 202683437 | LIRF→EGBB | U | 14,096 | −159 | 01-09 10:50 | – | – | 01-09 15:21 | 16,262 | – |
| 213145779 | LIRF→KJFK | U | 12,942 | −861 | 07-11 04:10 | – | – | 07-11 08:35 | 15,902 | – |
| 213141547 | LIRF→LEBL | U | 12,456 | −799 | 07-02 06:35 | – | – | 07-02 12:06 | 19,863 | – |
| 202690553 | LIRF→EPGD | U | 12,332 | 0 | 01-25 21:15 | – | – | 01-26 01:05 | 13,859 | – |
| 213141229 | LIRF→LIRN | U | 12,187 | 0 | 07-02 10:15 | – | – | 07-02 13:59 | 13,496 | – |
| 213141242 | LIRF→LIRN | U | 12,120 | 0 | 07-02 10:30 | – | – | 07-02 14:10 | 13,203 | – |
| 212756714 | LFPG→EDDM | M | 11,901 | +142 | 07-06 10:55 | 07-06 10:55 | 07-06 11:00 | 07-06 14:42 | 13,620 | 13,620 / 13,620 / 13,313 |
| 213142729 | LIRF→LICB | U | 11,830 | 0 | 07-05 09:40 | – | – | 07-05 13:17 | 13,025 | – |
| 213147579 | LIRF→OMSJ | U | 11,521 | −1,013 | 07-20 11:35 | – | – | 07-20 16:09 | 16,495 | – |
| 213153533 | LIRF→CYYZ | U | 11,061 | −792 | 07-16 10:25 | – | – | 07-16 14:34 | 14,945 | – |
| 202681908 | LIRF→EHAM | U | 11,054 | −935 | 01-06 08:55 | – | – | 01-06 13:51 | 17,816 | – |

### Checking the note's item D

The note is `HANDOFF_PROMPT.md` item D on `origin/claude/peaceful-babbage-md9l2x`.

What the note says: about 27 unmatched rows at EHAM, EDDF and LFPG have **ms (MVT − SCHED)** of 30k–96k s, and the
model currently predicts about 2,000 s for them. The 30k–96k range describes MVT − SCHED, not the predictions.

Measured:
- **29** non-LIRF unmatched scored rows have `mvt_minus_sched` in [30,000, 96,500]. By airport: EHAM 18, EDDF 4,
  LFPG 2, LEBL 2, EDDM 1, EGLL 1, LSZH 1.
- Their v33 predictions range from **659 to 2,987 s**.
- No non-LIRF unmatched row has a v33 prediction above 10,800 s.

So the reading "about 27 unmatched rows have predictions of 30k–96k s" is not what the note says, and it is not
what the data shows for non-LIRF rows. The unmatched rows whose v33 prediction falls in 30k–96k number **20, and all
of them are LIRF**. Their predictions are about equal to their own MVT − SCHED.

For comparison, the OOF has 25 non-LIRF unmatched rows with ms > 30,000. Their labels range from 549 to 3,849
(median 1,080), none is above 10,800, and their OOF predictions go up to 4,808.

## 2. Training labels (2025, DEP, `feat_train`)

| subset | n | mean | min | q50 | q99 | q99.9 | q99.99 | max | > 3,600 | > 7,200 | > 10,800 | > 30,000 | < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 2,085,047 | 991.2 | −12 | 912 | 2,339 | 4,501 | 10,690 | 131,167 | 4,126 | 584 | 202 | 54 | 369 |
| matched | 2,062,577 | 986.9 | −12 | 911 | 2,303 | 3,959 | 6,290 | 87,002 | 3,039 | 103 | 16 | 1 | 129 |
| unmatched | 22,470 | 1,384.4 | −12 | 951 | 10,211 | 53,410 | 88,132 | 131,167 | 1,087 | 481 | 186 | 53 | 240 |
| unmatched LIRF | 1,488 | 6,530.6 | 62 | 3,953 | 59,641 | 88,392 | 131,167 | 131,167 | 745 | 439 | 180 | 50 | 0 |
| unmatched non-LIRF | 20,982 | 1,019.4 | −12 | 912 | 4,667 | 8,220 | 58,206 | 87,341 | 342 | 42 | 6 | 3 | 240 |

Where the 202 labels above 10,800 s fall, by airport and lane:

| airport | unmatched | matched |
|---|---:|---:|
| LIRF | 180 | 8 |
| LFPG | 2 | 4 |
| EGLL | 1 | 3 |
| EHAM | 1 | 1 |
| LSZH | 2 | 0 |

By month, the 202 rows are spread over all 12 months, from 6 (Nov) to 50 (Jul).

The raw training files have the same DEP counts. ARR rows, which are not used, have 47 rows above 10,800 s.

**How the pipeline treats these rows: no label is clipped, capped or excluded from fitting.**
- The only filter on the fit rows drops null or non-finite labels, at train.py:1394:
  `train_f = train_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())`
- Fit targets:
  - Model A (anchored): `LABEL − mvt_minus_aobt3`, at train.py:1457–1460. The plain-y member fits `LABEL`.
  - Model B: `LABEL − sched_reparam_shift`, at train.py:910–912. The shift is `mvt_minus_sched` at LIRF and 0
    elsewhere.
- A label window is applied only inside two label-derived encodings, not to the fit:
  - `reference_table`, at train.py:515:
    `lab = train_dep.filter(pl.col(LABEL).is_not_null() & (pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))`
  - `agg_ref_iqr`, at train.py:667, with the same window.
- Two operations apply to the output, not the label:
  - A zero floor at train.py:1577: `.fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0)`
  - The LOBT window `[max(mvt_minus_lobt − 3606, 0), mvt_minus_lobt + 3606]` on matched rows only (train.py:1526–1561,
    `lobt_bound: true` in every member config). Unmatched rows have no window. The train.py:277–279 comment says:
    "That is a different operation from capping the label, which five board measurements refute".
- The ADS-B residual clips its own target and its correction, not the base prediction. This is in
  `scripts/adsb_learn_cv.py:32–33` (`CLIP_Y = 3600.0`, `CLIP_C = 1800.0`), applied in `adsb_learn_apply.py:54,78`.

## 3. Large values in the OOF of the current base (`hybrid_blend_l2_pa_opt`)

These figures are for 672,428 scored rows. The OOF RMSE is 268.320 and the SSE is 4.8412e10. Matched RMSE is
205.58. Unmatched RMSE is 1,558.42 on 8,378 rows.

| row set | n | matched / unmatched | LIRF | label also > 10,800 | pred also > 10,800 | SSE | share of SSE | RMSE without these rows |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **pred > 10,800** | **68** | 2 / 66 | 67 | **61** (64 > 3,600) | – | 4.115e9 | **8.50 %** | 256.68 |
| **label > 10,800** | **78** | 5 / 73 | 71 | – | **61** | 1.398e10 | **28.88 %** | 226.30 |
| both > 10,800 | 61 | 2 / 59 | 60 | 61 | 61 | 3.393e9 | 7.01 % | 258.76 |
| pred > 10,800, label ≤ 10,800 | 7 | 0 / 7 | 7 | 0 | 7 | 7.217e8 | 1.49 % | 266.31 |
| label > 10,800, pred ≤ 10,800 | 17 | 3 / 14 | 11 | 17 | 0 | 1.059e10 | 21.87 % | 237.17 |
| either > 10,800 | 85 | 5 / 80 | 78 | 78 | 68 | 1.470e10 | 30.37 % | 223.91 |

Notes on the table:
- The 7 rows with a high prediction but a lower label are all LIRF unmatched. Their labels are 723, 959, 1,377,
  1,926, 9,669, 10,391 and 10,799.
- By lane, the matched rows contribute 0.02 % of SSE (pred > 10,800) and 0.13 % (label > 10,800).
- One matched LIRF OOF row repeats the "+1 day" pattern seen in the test row 213147416: y = 87,002 and pred 84,616.

The same measures at other thresholds:

| threshold | pred above it: n, of which label also above, SSE share | label above it: n, of which pred also above, SSE share |
|---|---|---|
| 3,600 | 852, 623, 15.74 % | 1,464, 623, 39.31 % |
| 7,200 | 163, 132, 11.22 % | 210, 132, 31.20 % |
| 30,000 | 24, 24, 6.78 % | 26, 24, 27.84 % |

LIRF unmatched rows in the OOF: 537 rows, 69 with label > 10,800, 102 with ms > 10,800 and 66 with a prediction
above 10,800.

One counterfactual, a measurement only and not a proposal: capping every OOF prediction at 10,800 raises the OOF
RMSE from 268.32 to 417.13 (ΔSSE +6.86e10).

## 4. Existing cap or clip rules in code and docs

The grep searched `src/`, `scripts/` and `docs/` for "clip", "cap", "10800", "10_800" and "10,800".

**Code:**
- `src/prc/validate_submission.py:19`: `MAX_PLAUSIBLE_SECONDS = 3 * 3600  # sanity bound, not a clipping rule`.
  At :90–91 it only appends the warning `f"max prediction {mx} exceeds {MAX_PLAUSIBLE_SECONDS}s; check outliers"`.
- `src/prc/train.py:1577`: a zero floor, `clip(lower_bound=0.0)`.
- `src/prc/train.py:1526–1561`: the LOBT window, matched rows only.
- `src/prc/train.py:277–279`: "a different operation from capping the label, which five board measurements refute".
- ADS-B residual: `CLIP_Y = 3600.0`, `CLIP_C = 1800.0` (`scripts/adsb_learn_cv.py:32–33`). These clip the residual
  target and the correction only.
- There is **no upper cap on predictions anywhere** in `src/` or in the production scripts.

**Docs:**
- `docs/investigation_2026-09-16/results/tail-postprocessing-cap-shrink.md` §4, run on the pre-reparam OOF
  `oof_main-48c19bc830`:
  - "(a) A fixed cap on unmatched predictions improves RMSE | REFUTED (on this OOF) | Every cap ≤ 43,200 raises
    unmatched RMSE … and total RMSE (334.6 → 452.6 / 433.6 / 404.6 / 335.4)"
  - "(e) Capping matched rows improves RMSE | REFUTED"
  - "Hypothesis as a whole … REFUTED for cap / shrink applied to all unmatched rows or to matched rows; CONFIRMED only
    in the restricted form 'cap non-LIRF unmatched rows', worth −7.5 total RMSE on this OOF"
  - §3.6: a non-LIRF unmatched cap at 3,600 gives 334.58 → 327.05 on 53 rows (49 helped / 4 hurt).
- `docs/competitor_research_2026-09-21/plans/P1-unmatched-lane.md` Stage 1: "the 3,600 cap is closed — the
  reparameterisation already collected it". In v10, non-LIRF unmatched predictions above 7,200 numbered 0 and the
  max was 4,689. (In v33, the non-LIRF unmatched max is ≤ 2,987 for the ms > 30k rows, and no non-LIRF unmatched
  prediction exceeds 10,800.)
- `docs/competitor_research_2026-09-21/SUMMARY.md` §F (settled negatives): "Clean or cap the **label** | drop
  >120 min +24.1; drop >60 min +36.3; cap 7200 s +77.6; cap 4500 s +96.1; drop 17 rows >6 h +13.7 …; Phoenix-Ops
  label cap 434.1 vs 382.3".
- `docs/competitor_research_2026-09-21/plans/P7-model-capacity.md:308`: "**Clip the fitting residual. Never the
  label.**" This is a plan and is not in the production code.
- `docs/competitor_research_2026-09-21/plans/P6-per-airport-upper-bound.md`: a plan proposing a per-airport bound
  on 12 matched LEMD/LEBL rows. It is not in the code.
- `docs/competitor_research_2026-09-21/plans/P6-REVIEW.md:308`: the > 10,800 s check "is a **warning only**".
- `docs/competitor_research_2026-09-21/results/24-repo-tail-sweep.md:390`: a competitor repo post-processes with
  `np.clip(pred, 60, 10800)`. This is external code and not ours.
