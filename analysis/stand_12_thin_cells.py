"""RMSE of each variant on the thin stand x runway rows (< 30 labelled rows in the months the fold's reference
table was fitted on: all 2025 months except the scored month and the inner early-stopping month)."""

from __future__ import annotations

import glob
import sys

import numpy as np
import polars as pl

from stand_03_model import paired, rmse
from stand_common import SCRATCH

tag = sys.argv[1] if len(sys.argv) > 1 else "_fast"
variants = sys.argv[2:] or ["occ", "grp", "occ_grp", "geo", "all"]
b = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
               for f in sorted(glob.glob(f"{SCRATCH}/base_*.parquet"))]).with_columns(
    pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
thin_ids = []
for m, inner in ((1, 12), (7, 6)):
    cnt = b.filter(~pl.col("m").is_in([m, inner])).group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).len()
    v = b.filter(pl.col("m") == m).join(cnt, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
    thin_ids.append(v.filter(pl.col("len").fill_null(0) < 30).select("MVT_ID_mvt"))
thin = pl.concat(thin_ids).with_columns(pl.lit(True).alias("thin"))
a = pl.read_parquet(f"{SCRATCH}/oof_base{tag}.parquet").join(thin, on="MVT_ID_mvt", how="left", maintain_order="left")
t = a["thin"].fill_null(False).to_numpy()
y = a["TAXITIME_SEC_mvt"].to_numpy().astype(float)
c = (y >= 30) & (y <= 7200)
mt = a["mvt_minus_aobt3"].is_not_null().to_numpy()
print("thin rows", t.sum(), "share", t.mean())
rows = []
for v in variants:
    o = a.select("MVT_ID_mvt").join(pl.read_parquet(f"{SCRATCH}/oof_{v}{tag}.parquet").select(["MVT_ID_mvt", "pred"]),
                                    on="MVT_ID_mvt", how="left", maintain_order="left")
    pa, pb = a["pred"].to_numpy(), o["pred"].to_numpy()
    for seg, mask in (("thin", t), ("thin clean", t & c), ("thin matched", t & mt), ("not thin, matched", (~t) & mt)):
        d, ci, p = paired(y[mask], pa[mask], pb[mask], n=500)
        rows.append((v, seg, int(mask.sum()), rmse(y[mask], pa[mask]), rmse(y[mask], pb[mask]), d, ci[0], ci[1], p))
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows, schema=["variant", "segment", "n", "base", "new", "diff", "ci_lo", "ci_hi", "P"], orient="row")
      .with_columns([pl.col(x).round(2) for x in ("base", "new", "diff", "ci_lo", "ci_hi")]))
