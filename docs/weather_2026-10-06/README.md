# Weather beyond the base's weather columns (2026-10-06): no gain

The base members already use hourly METAR (`use_weather: true`). Cached IEM METAR (`data/ref/metar`) was reused.
Snow-hour share of January departures 2025 → 2026: LSZH 1.1 → 9.4 %, EDDF 1.3 → 6.6, LTFM 0 → 8.6, EHAM 1.5 → 5.4.

A residual model on weather flags, leave-one-month-out on the 7-month OOF: −10 [−26, +5] s² per row. On the 12-month
OOF: −32 [−79, +8]. With continuous weather: −126. No positive estimate in any arm, and the snow-hour evidence comes
from 3–17 days per airport.

LTFM, the one big 2025 snow event (Feb 2025): on matched rows, y − (MVT − AOBT_3) moves from −286 s on non-snow
hours to +8 s on snow hours (+294). For January 2026, v40's prediction − (MVT − AOBT_3) moves from −215 to +145
(+360), so the production base already applies the snow uplift. Closed.
Scripts: `analysis/local_wx_build.py`, `analysis/local_wx_resid.py`.
