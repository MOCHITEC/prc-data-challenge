"""Paired-bootstrap table of stand variants vs base on jan+jul OOF, split into NM-matched / unmatched rows and
'bound active' (same-stand in-block < 30 min before take-off). Usage: python stand_11_segment_table.py TAG v1 v2 ..."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from stand_03_model import paired, rmse
from stand_common import SCRATCH

tag = sys.argv[1] if len(sys.argv) > 1 else "_fast"
variants = sys.argv[2:] or ["occ", "nbr", "geo", "all"]
a = pl.read_parquet(f"{SCRATCH}/oof_base{tag}.parquet")
occ = pl.read_parquet(f"{SCRATCH}/stand_feats.parquet").select(["MVT_ID_mvt", "g_arr_prev1"])
a = a.join(occ, on="MVT_ID_mvt", how="left", maintain_order="left")
y = a["TAXITIME_SEC_mvt"].to_numpy().astype(float)
pa = a["pred"].to_numpy()
m = a["mvt_minus_aobt3"].is_not_null().to_numpy()
c = (y >= 30) & (y <= 7200)
act = (a["g_arr_prev1"].fill_null(1e9) < 1800).to_numpy()
rows = []
for v in variants:
    b = pl.read_parquet(f"{SCRATCH}/oof_{v}{tag}.parquet").select(["MVT_ID_mvt", "pred"])
    b = a.select("MVT_ID_mvt").join(b, on="MVT_ID_mvt", how="left", maintain_order="left")
    pb = b["pred"].to_numpy()
    for seg, mask in (("all", np.ones_like(m)), ("clean", c), ("matched", m), ("matched & bound", m & act),
                      ("unmatched", ~m), ("unmatched clean", (~m) & c)):
        d, ci, p = paired(y[mask], pa[mask], pb[mask], n=500)
        rows.append((v, seg, int(mask.sum()), rmse(y[mask], pa[mask]), rmse(y[mask], pb[mask]), d, ci[0], ci[1], p))
    pc = np.where(m, pb, pa)
    d, ci, p = paired(y, pa, pc, n=500)
    rows.append((v, "all, new only on matched", len(y), rmse(y, pa), rmse(y, pc), d, ci[0], ci[1], p))
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows, schema=["variant", "segment", "n", "base", "new", "diff", "ci_lo", "ci_hi", "P(better)"], orient="row")
      .with_columns([pl.col(x).round(2) for x in ("base", "new", "diff", "ci_lo", "ci_hi")]))
