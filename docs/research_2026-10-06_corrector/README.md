# Stage-2 corrector on the 4-month OOF — prototype (2026-10-06)

**Question**: does a residual corrector, in the style of smart-jigsaw's −4.05 s on the LB, add anything on top of a2o+L6
(main_flseq_osm_nolirf_l2_opt_l6-177174f530)?

**Setup**
- Target: r = y − oof, clipped to ±1800 s. Matched rows only, 664,050 rows.
- Validation: leave-one-month-out over the 4 OOF months (Jan, Jul, Nov, Dec).
- Inputs:
  - oof and the five NM/schedule clocks;
  - distances from oof to each clock (dist_plano) and their minimum;
  - neighbour context and weather;
  - leave-month-out cell means of r (airport×stand×runway, airport×hour, airport×type), nested on the training side;
  - categoricals.
- Script: `proto.py`. `proto_l2.py` is the same script with an L2 loss, 31 leaves, min_data 500 and 300 rounds.

**Result: worse at every strength and in every month.**

| loss | shrink | dRMSE [CI] | Jan / Jul / Nov / Dec |
|---|---|---|---|
| Huber (α=300) | 1.0 | +2.30 [+2.04, +2.53] | +2.67 / +3.47 / +1.05 / +1.58 |
| Huber | 0.5 | +0.52 [+0.39, +0.64] | +0.66 / +1.10 / −0.06 / +0.17 |
| L2 | 1.0 | +1.58 [+1.35, +1.80] | +1.52 / +2.16 / +0.69 / +1.73 |
| L2 | 0.5 | +0.41 [+0.30, +0.52] | +0.37 / +0.67 / +0.02 / +0.46 |

**Reading**
- The production member already uses 2000 rounds with every one of these inputs (dist_plano aside). What is left of
  its residual is not learnable across seasons from 3 training months: Jan is corrected from Jul/Nov/Dec and gets
  worse.
- smart-jigsaw had 10 cross-fitted months and a weaker base (254 on the LB).
- Doing it properly would mean block cross-fitting every member on all 12 months: about 6 refits per member, more
  than 10 h on this machine. The prototype gives no reason to expect it to pay. **Not pursued.**
