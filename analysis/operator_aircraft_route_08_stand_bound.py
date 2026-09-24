"""operator_aircraft_route 08: the stand re-occupation lower bound as a post-processing rule on OOF predictions.

If another aircraft blocked in at our stand after our NM off-block (AOBT_3) and before our take-off, we must have
left the stand before that in-block, so BLOCK <= inblock_next  =>  y >= stand_lb = MVT - inblock_next.
Rule: pred' = max(pred, stand_lb - margin) when stand_lb <= cap (the bound is invalid above ~1 h, see output). Evaluated on the OOF of 06c (jan+jul).

usage: python operator_aircraft_route_08_stand_bound.py [arm]   (default arm: base)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from operator_aircraft_route_common import INTERIM, md, paired_boot


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else "base"
    oof = pl.read_parquet(os.path.join(INTERIM, f"oar_oof_{arm}.parquet"))
    nf = pl.read_parquet(os.path.join(INTERIM, "oar_newfeats.parquet"), columns=["MVT_ID_mvt", "stand_lb", "ta_gap",
                                                                                   "ta_aobt3_minus_inblock"])
    o = oof.join(nf, on="MVT_ID_mvt", how="left")
    y, p = o["y"].to_numpy(), o["pred"].to_numpy()
    lb = o["stand_lb"].to_numpy().astype(float)
    has = np.isfinite(lb)
    print(f"## 下限ルール（arm={arm}, jan+jul OOF {len(y):,} 行）\n")
    print(f"下限あり行: {has.mean():.4f}; 真値が下限を満たす率 y>=lb: {(y[has] >= lb[has]).mean():.4f}; "
          f"予測が下限を下回る率 pred<lb: {(p[has] < lb[has]).mean():.4f}\n")
    rows = []
    for cap, margin in ((1e9, 0), (3600, 0), (2700, 0), (2700, 60), (1800, 0)):
        ok = has & (lb <= cap)
        q = np.where(ok, np.maximum(p, lb - margin), p)
        d, lo, hi = paired_boot(y, p, q, n=300)
        chg = ok & (q != p)
        rows.append((cap / 60 if cap < 1e8 else float("inf"), margin, int(chg.sum()), float(np.sqrt(np.mean((y - q) ** 2))), d, lo, hi,
                     float(np.sqrt(np.mean((y[chg] - p[chg]) ** 2))) if chg.any() else float("nan"),
                     float(np.sqrt(np.mean((y[chg] - q[chg]) ** 2))) if chg.any() else float("nan")))
    print(md(pl.DataFrame(rows, schema=["lb_cap_min", "margin_s", "rows_changed", "rmse_after", "delta", "ci_lo", "ci_hi",
                                        "rmse_changed_before", "rmse_changed_after"], orient="row"), "{:,.2f}"))
    print(f"\n基準 RMSE = {np.sqrt(np.mean((y - p) ** 2)):.2f}\n")


if __name__ == "__main__":
    main()
