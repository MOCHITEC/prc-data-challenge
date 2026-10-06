"""Build a candidate = BASE submission + weighted member changes on NM-matched rows.

    .venv/bin/python scripts/build_member_delta.py BASE OUT old1:new1:w1 [old2:new2:w2 ...]

Each oldK/newK is a ranking-prediction file written by `prc.pipeline --skip-cv` (outputs/<name>.parquet), and wK
is that member's weight in the matched blend of BASE. For the v35-v40 chain the matched blend is
((0.5 a2o + 0.5 pao) + p2o + c2 + m4) / 4, so a2o and pao weigh 1/8 each and p2o 1/4.

The change sum_k wK (newK - oldK) is added on matched rows only; unmatched rows do not move by construction.
Approximation: the ADS-B residual and the overlays on top of BASE are not refitted (|delta| is a few seconds).
The output passes the submission gate before it is written."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

from prc.validate_submission import validate_files

TPL = "data/raw/submitting.parquet"
base_f, out_f, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
tpl = pl.read_parquet(TPL)
d = pl.read_parquet(base_f)
assert d.height == tpl.height and (d["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
matched = pl.read_parquet("data/work/feat_rank.parquet", columns=["MVT_ID_mvt", "mvt_minus_aobt3"]).select(
    "MVT_ID_mvt", pl.col("mvt_minus_aobt3").is_not_null().alias("matched")
)
d = d.join(matched, on="MVT_ID_mvt", how="left", maintain_order="left")
assert d["matched"].null_count() == 0
delta = np.zeros(d.height)
for k, spec in enumerate(specs):
    old, new, w = spec.split(":")
    w = float(eval(w))  # weights like 1/8
    o = pl.read_parquet(old)
    n = pl.read_parquet(new)
    for f in (o, n):
        assert f.height == tpl.height and (f["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all(), f
    dk = (n["TAXITIME_SEC_mvt"].cast(pl.Float64) - o["TAXITIME_SEC_mvt"].cast(pl.Float64)).to_numpy()
    m = d["matched"].to_numpy()
    print(f"{Path(new).stem}: weight {w:.4f}; member change on matched RMS {np.sqrt((dk[m] ** 2).mean()):.2f} s")
    delta += w * dk
m = d["matched"].to_numpy()
delta[~m] = 0.0
new = np.clip(np.round(d["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy() + delta), 0, None).astype(np.int32)
out = pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "TAXITIME_SEC_mvt": new})
assert out.schema == tpl.schema
assert (out["TAXITIME_SEC_mvt"].to_numpy()[~m] == d["TAXITIME_SEC_mvt"].to_numpy()[~m]).all()
out.write_parquet(out_f)
print(
    f"rows {d.height:,}; matched {m.sum():,}; changed {(new != d['TAXITIME_SEC_mvt'].to_numpy()).sum():,}; "
    f"delta on matched mean {delta[m].mean():+.2f} RMS {np.sqrt((delta[m] ** 2).mean()):.2f} "
    f"p1 {np.percentile(delta[m], 1):+.1f} p99 {np.percentile(delta[m], 99):+.1f}"
)
res = validate_files(Path(out_f), Path(TPL))
print("gate:", "PASS" if res.ok else "FAIL", res.warnings)
res.raise_if_failed()
