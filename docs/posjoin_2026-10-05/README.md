# Positional ADS-B join for unmatched departures (2026-10-05)

Candidate overlay on v33 (LB 250.9496). **Not submitted.** The row list is in `replacements_v33.csv` (MVT_ID_mvt, v33, new, airport, day, rule).

The overlay replaces 430 of 344,841 rows. The squared change it makes is 1,623 MSE in board units. The
expected board gain is about 1,200 to 1,950 MSE, which takes v33 to roughly **247 to 248.5**. The gain is
unmeasured, because the board is the only scorer for 2026.

## Why these rows

* Unmatched 2026 departures have no NM record, so they have no `CALLSIGN_flt`. As a result, the callsign
  ADS-B join, and with it the learned ADS-B residual, never reaches them: the learned residual is 0 on all
  430 rows.
* `FLIGHT_mvt` is in IATA form (`KL0965`) and the transponder callsign is in ICAO form (`KLM965`, `KLM47H`).
  The two almost never match.
* EHAM 2026-01-02..07 was a snow and de-icing disruption with no counterpart anywhere in 2025.
  * On 01-04 the airport-day median of `MVT − AOBT_3` was 2,352 s. The 2025 maximum was 2,640 s, at LTFM.
  * On that day 123 of 336 departures were unmatched.
  * v33 predicts about 2,400–2,500 s for those flights. The ADS-B ground tracks give a median of
    3,350–3,750 s.

## Method (`analysis/local_adsb_posjoin.py`)

For each departure without a callsign join, the script takes the aircraft whose ground→air transition at the
same airport lies within ±90 s of MVT_TIME. MVT_TIME is first shifted by the airport's median ADS-B take-off
offset (p10–p90 within about ±40 s). Two more conditions apply:

* the take-off heading must agree with `RUNWAY_mvt` to within 25°;
* the aircraft type must not contradict `AIRCRAFT_TYPE_mvt`.

Take-offs already claimed by the callsign join are excluded. The script then runs
`adsb_stage1_feasibility.stand_departure` on that aircraft, unchanged.

On the 430 target rows:

* 429 of 430 are type-matched; the one remaining row has no type;
* none has a runner-up take-off within 30 s;
* the median |Δt| between take-off and MVT is 7 s.

## Rules (`analysis/local_posjoin_overlay.py`), LIRF excluded (schedule-fill labels)

| rows | condition | new value | 2025 check against the label |
|---|---|---|---|
| 332 | push signature (`push ≥ 0.5`) and no later stop ≥ 900 s | `MVT − t_move` | 225 rows: RMSE 108 s, against 357 s for the model |
| 98 | push signature and a later stop ≥ 900 s | `base + 0.75·(a − base)`; `a` = `MVT − end of stop` if the stop began < 100 m from the first moved fix, else `MVT − t_move` | 263 rows: 84 % within 5 min of BLOCK, RMSE 765 s |

### Why the long-stop rows need their own rule

* In 2025, flights whose stop ran 30 minutes or more had BLOCK more than 15 minutes after the ADS-B push on
  73–90 % of rows. Those are tows and repositioning.
* Where the stop is far from the stand (median 1.4 km), BLOCK sits at the push, as with de-icing or a queue.
* Where the stop starts right at the first moved fix (median 0.6 m), BLOCK sits at the end of the stop.

`analysis/local_dwell_end.py` computes the start and end of the stop.

`AOBT_3` is **not** a usable truth proxy for long-stop flights. In 2025 it came more than 15 minutes after
BLOCK on 63 % of rows with a 15–30 minute stop, so the 2026 matched-flight check of the long-stop rule is
uninformative.

### The w = 0.75 weight

* With the 2025 noise level (765 s), the expected gain is 1,213 MSE at w = 0.75 and 1,227 at w = 1.
* If the snow-day noise were twice that, the expected gain is about 500 at w = 0.75 and about 280 at w = 1.

## Not done

* A positional join for the matched lane is not needed, because the callsign join already covers it.
* Rows without a push signature are left alone: in 2025, raw ADS-B was worse than the model on them.

## Other findings from this session (all measured locally)

* **ADS-B ground coverage across 2025** (`local_adsb_coverage_shift.py` plus a sample of the 15th of each
  month). Nov–Dec 2025 coverage resembles Jan 2026:

  | airport | Jan 2025 | Nov–Dec 2025 |
  |---|---|---|
  | LFPG | 0 | 0.61–0.73 |
  | LIRF | 0 | 0.87–0.89 |
  | LEMD | 0 | 0.98 |
  | EGLL | 0.85 | 0.81–0.87 |

  The learned residual is trained on Jan and Jul 2025 only. Training it on more months is the next lever. On
  the board, v21's gain was 1.04× its CV gain, against the 1.35× that a reweighting to the 2026 coverage mix
  predicted.
* **Leave-one-airport-out transfer of the learned residual** (`local_adsb_loao.py`): the transfer ratio is
  0.27 to 0.90 at eight airports. It is negative only at LFPG (−0.38), where the in-airport gain is tiny
  anyway. LTFM was skipped (38 ADS-B rows).
* **LIRF schedule-copy labels** (`local_lirf_copy_clf.py`): 15–21 % of labels in every month are schedule
  copies.
  * A classifier separates them with AUC 0.87.
  * The cross-month blend gain is only −0.4 to −0.7 s; the oracle bound is −7.5 s.

Inputs live in `$SCRATCH` (the ADS-B extracts are ODbL: "Contains data from adsb.lol, made available under the
ODbL 1.0").
