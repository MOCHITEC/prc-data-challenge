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
