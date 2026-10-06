# LIRF: the "one day + taxi" label class on long-delayed unmatched departures (2026-10-06)

## Finding

LIRF departures without NM AOBT_3 that take off more than ~14 h after schedule carry one of two labels in 2025:

| MVT − SCHED | schedule copy (y = MVT − SCHED) | one day + taxi (y = 86 400 + T) | normal taxi |
|---|---|---|---|
| 8.5–14 h (30 000–50 400 s) | 28 | 0 | 0 |
| 14–15.5 h (50 400–56 000 s) | 4 | 1 | 0 |
| ≥ 15.5 h | 5 | 10 | 0 |

The one-day rows (T median 1 080 s, sd 400) are spread over 8 months of 2025 (Jan, Mar, Jun, Jul ×4, Sep, Nov ×2,
Dec), so this is a standing recording habit, not an incident. Mechanism: the off-block clock time is entered and placed
on the date nearest the schedule, so a delay beyond ~13.5 h lands one day early. No other airport shows it (all
long-delayed unmatched departures elsewhere have a normal taxi); matched LIRF rows never show it; arrivals never show it.
Two public repos report the same class (kind-mango: "LIRF 24 h class, predicted as 86 400 + 1 150 s"; joris-decombe:
"LIRF orphans 14–26 h late are either at schedule or one day plus a normal taxi", −5.2 s in their validation).

v40 predicts all six 2026 rows of this kind as schedule copies (y ≈ MVT − SCHED + ~1 500).

## Candidate C7 (not submitted)

`analysis/local_lirf_wrap.py apply` replaces the prediction on LIRF unmatched rows with MVT − SCHED ≥ 50 400 s by the
expectation q · (86 400 k + T) + (1 − q) · (MVT − SCHED), q = (wraps + 1) / (rows + 2) per band (0.29 below 56 000 s,
0.65 above). Six rows change (one, ms 87 484, is unchanged in effect).

| 2026 outcome | gain vs v40 (MSE, full ranking set) |
|---|---|
| expected at 2025 rates | **+3 543** (≈ −7.5 s at 238.8) |
| all six are copies | −3 795 |
| all six are one-day | +8 332 |

2025 leave-one-out check of the blend (20 rows, p and T from the other rows): positive, 6.8e7 s² per row.
Output: `outputs/c7_v40_lirfwrap.parquet` (v40 + this blend only).

## Follow-up (same day): mechanism refined, two more LIRF label classes

- **The off-block is put on SCHED's UTC date** (1 486 of 1 488 unmatched LIRF departures in 2025), not the nearest
  date. A one-day label therefore needs the off-block to cross UTC midnight relative to SCHED, and the re-dated
  block must land between about SCHED − 10.4 h and SCHED + 1.8 h (where it gets "accepted"). Per 2026 row:
  TAY9LA does not cross midnight, so q 0.29 → 0.05. NOSOS431 lands at SCHED + 6.7 h, so q 0.65 → 0.15. The other
  three stay at about 0.65. Expected about +600 MSE over C7. (`analysis/local_lirf_wrap_stress.py`)
- **Time-zone class:** BLOCK = true off-block − Rome UTC offset, so y = 3 600 (winter) or 7 200 (summer) + taxi.
  It flips exactly on the DST dates and is 37 rows a year, carried by a few operators (BEL 13/14, AEZ 4/6, AEA 3/3,
  LZB 2/2 among unmatched rows with ms < 1 h). Blend q = tz / (n + 1) per prefix. Leave-one-month-out on the OOF:
  +117 MSE, 6/6 months positive. 6 rows in 2026.
- **Copy band:** unmatched LIRF rows with a delay of 7–13.5 h are all schedule copies in 2025 (34/34). Predict
  MVT − SCHED. Leave-one-month-out: +72 MSE, no month negative. 17 rows in 2026.
- **LTFM "copies"** are real on-time pushbacks (no spike at BLOCK = SCHED compared with ±60 s). LEBL shows a 2×
  spike, but a recalibration is not robust (2–3 of 7 months negative).
- **LFPG/EGLL unmatched:** 14.6 of LFPG's 15.6 % OOF SSE share is two easyJet rows with a one-day-early BLOCK.
  These are unpredictable. No other label artefact. An LFPG-only small residual is worth +7 to +19 MSE only and is
  parked. (`analysis/local_lfpg_unm.py`)

Candidate **C8** = v40 + revised C7 + time-zone class + copy band: 29 rows changed,
`outputs/c8_v40_lirf_wrap_tz_copy.parquet` (`analysis/local_lirf_artefacts_apply.py`).
