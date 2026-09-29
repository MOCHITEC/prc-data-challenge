# ADS-B pushback override: full ingest of the 62 scored days, and a submission candidate (2026-09-29)

Sibling note to `adsb-oos-validation.md` (the out-of-sample GO this acts on), `adsb-gain-estimate.md` and
`adsb-stage1-feasibility.md`. Nothing here was submitted. Nothing was written to `outputs/` or `src/`, and nothing
was fitted.

## 0. Pre-registration (written before any new 2026 day was ingested and before the EGLL check was run)

At the time this section was written, the scratch dir already held the five 2026 census days of the validation
(01-19, 01-28, 07-08, 07-20, 07-29). None of the other 57 scored days had been downloaded. The EGLL
leave-one-day-out check below had not been run. The only thing looked at for the 57 days was a HEAD of each
release: all 62 days have one, 2.20–4.28 GB each, 213.2 GB in total `[H]`.

### 0.1 Production settings, fixed here

| item | value |
|---|---|
| detection | stage-1 rules, `scripts/adsb_stage1_feasibility.py` unchanged (last changed in `ae98d1f`; identical at `9f29f62` and `b807fa8`). Pushback gate share ≥ 0.5, 15-min dwell guard, 25 m stand radius, 10-min silence. Scope is the stage-1 scope: departures with `MVT_TIME_UTC_mvt` in [D 02:00, D+1 00:00) UTC, read from archive day D. Departures between 00:00 and 02:00 UTC keep the base prediction, and their count is reported. |
| join | `CALLSIGN_flt` at `ADEP_mvt`. A row qualifies only if stage-1's `join == "cs"`. The stage-1 fallback to `FLIGHT_mvt` is not used: it produced 1 of 9,269 covered lane-A rows over the nine days on disk, and none at a GO airport. On the scored set, `CALLSIGN_flt` is present exactly when `AOBT_3_flt` is (lane A), so the override is lane A only, which is the population the gain was validated on. |
| guard | `y_a = MVT − t_push + b`. Use `y_a` unless `|base − y_a| > 900 s`, in which case keep the base. |
| airports | **EDDM, EHAM, LEBL, LSZH.** LEMD is out. EGLL is conditional (0.2). Every other airport keeps the base. |
| bias `b` | per airport, the median of (`t_push` − `BLOCK`) over the covered rows (lane A, `join == "cs"`, gated pushback) of **all four labelled days**: 2025-07-21, 2025-07-13, 2025-01-22, 2025-11-12. Pooled over rows, not averaged over days. |
| base prediction | `outputs/main_gap.parquet` (v14, board 279.5234), read only. |
| candidate value | `round(y_a)` cast to Int32, the template's dtype. Every other row is the base value, bit for bit. |
| missing days | if a day cannot be fetched, its flights keep the base, and the day is listed. |

What the override reads is `t_push` (ADS-B), `MVT_TIME_UTC_mvt`, `ADEP_mvt`, the join key, the base prediction and
`b`. `BLOCK_TIME_UTC_mvt` is read only on the four labelled 2025 days, to estimate `b` and to score. On the scored
rows it is blank (0 of 344,841 non-null).

### 0.2 The EGLL rule, fixed before the check was run

- **Check.** Leave one labelled day out, over the four labelled days. For each held-out day `h`:
  - `b_EGLL(h)` is the median (`t_push` − `BLOCK`) over EGLL's covered rows on the other three days.
  - Apply the override with the 900 s guard against the production OOF (`experiments/oof_main_gap-62dd7c6e88`) on
    EGLL's covered rows of day `h`.
- **Gain.** The mean over the pooled held-out rows of (e_model² − e_override²), in s² per covered row. A tripped row
  counts as 0.
- **CI.** A 95 % row bootstrap, B = 2,000, the validation's primary CI. The airport × hour × day cluster CI and the
  per-day gains are reported but do not decide.
- **Include EGLL iff the pooled gain is ≥ 5,000 s² per covered row and the row-bootstrap CI's lower bound is > 0.**
  Otherwise EGLL is out.
- If EGLL is in, its production bias is the four-day median of 0.1, like every other airport.
- The same leave-one-day-out figure is reported for the four GO airports, for information only. Their inclusion is
  already decided `[OOS §4]`.
