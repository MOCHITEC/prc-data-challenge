# Member disagreement and stacking (2026-10-06): no gain

All three matched-lane members exist only for Jan+Jul 2025, so this is fit-on-one-month, test-on-the-other.

- **Error size:** member spread predicts the size of the error (RMSE 147 → 431 s from the lowest to the highest
  spread decile), but not its sign. On high-spread rows the plain mean beats every single member almost everywhere.
- **Per-airport and per-segment stacking weights:** −164 to −477 s² per row. They do not carry over between months.
- **Global reweight after the ADS-B residual:** about +13 s² per row (noise level).
- **Small LightGBM stacker:** negative.
- **2026:** per-member ranking predictions are not stored, so nothing can be applied.

Closed. Script: `analysis/local_member_stack.py`.
