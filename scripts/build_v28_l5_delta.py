"""Build the v28 candidate: v23 (LB 252.43) + one third of the L5 change of the anchored LightGBM member.

v23 = mean(anchored LGB, plain-y LGB, anchored CatBoost) on matched rows + learned ADS-B residual (scripts/
adsb_learn_apply.py). L5 (fe_tx_nm) was measured on the anchored member only (docs/research_2026-10-01 section 7),
so the change it makes to the 3-member mean is (anchored_l5 - anchored) / 3 on NM-matched rows; unmatched rows do
not move by construction. Approximation: the ADS-B residual is not refitted on the new base (|delta| is tens of s).

    .venv/bin/python scripts/build_v28_l5_delta.py [BASE_SUBMISSION] [OUT]

Used for v29 with BASE = v28 (hybrid_blend_l2 + learned ADS-B, LB 252.41): the same 1/3 delta, assuming the L5
change is additive to L2's (not measured; L5's f60 window overlaps L2's forward windows).
"""

import numpy as np
import polars as pl

from prc.validate_submission import validate_files

TPL = "data/raw/submitting.parquet"
import sys

# base submission (downloaded from the team bucket) and output; defaults reproduce the first candidate
V23 = sys.argv[1] if len(sys.argv) > 1 else "outputs/lb/resilient-hedgehog_v23.parquet"
A0 = "outputs/main_flseq_osm_nolirf.parquet"
A5 = "outputs/main_flseq_osm_nolirf_l5.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/v28_v23_plus_l5_delta.parquet"

tpl = pl.read_parquet(TPL)
v23 = pl.read_parquet(V23)
a0 = pl.read_parquet(A0).rename({"TAXITIME_SEC_mvt": "a0"})
a5 = pl.read_parquet(A5).rename({"TAXITIME_SEC_mvt": "a5"})
matched = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", "mvt_minus_aobt3"]).select(
    "MVT_ID_mvt", pl.col("mvt_minus_aobt3").is_not_null().alias("matched")
)
for f in (v23, a0, a5):
    assert f.height == tpl.height and (f["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
d = (
    v23.join(a0, on="MVT_ID_mvt", maintain_order="left")
    .join(a5, on="MVT_ID_mvt", maintain_order="left")
    .join(matched, on="MVT_ID_mvt", how="left", maintain_order="left")
)
assert d.height == tpl.height and d["matched"].null_count() == 0
delta = (
    pl.when(pl.col("matched")).then((pl.col("a5").cast(pl.Float64) - pl.col("a0").cast(pl.Float64)) / 3).otherwise(0.0)
)
d = d.with_columns(delta.alias("delta"))
d = d.with_columns(
    (pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) + pl.col("delta"))
    .round(0)
    .clip(lower_bound=0)
    .cast(pl.Int32)
    .alias("new")
)
um = d.filter(~pl.col("matched"))
assert (um["new"] == um["TAXITIME_SEC_mvt"]).all(), "unmatched rows must not move"
out = d.select("MVT_ID_mvt", pl.col("new").alias("TAXITIME_SEC_mvt"))
assert out.schema == tpl.schema
out.write_parquet(OUT)

dd = d["delta"].to_numpy()
m = d["matched"].to_numpy()
print(f"rows {d.height:,}; matched {m.sum():,}; changed {(d['new'] != d['TAXITIME_SEC_mvt']).sum():,}")
print(
    f"delta on matched: mean {dd[m].mean():+.2f} s, RMS {np.sqrt((dd[m] ** 2).mean()):.2f} s, "
    f"p1 {np.percentile(dd[m], 1):+.1f}, p99 {np.percentile(dd[m], 99):+.1f}, max|.| {np.abs(dd[m]).max():.1f}"
)
print(
    f"anchored member (no L5) vs v23 on matched rows: RMS diff "
    f"{np.sqrt(((d['a0'] - d['TAXITIME_SEC_mvt']).cast(pl.Float64).filter(pl.Series(m)) ** 2).mean()):.1f} s, "
    f"corr {np.corrcoef(d['a0'].to_numpy()[m], d['TAXITIME_SEC_mvt'].to_numpy()[m])[0, 1]:.4f}"
)
print(f"mean prediction: v23 {d['TAXITIME_SEC_mvt'].mean():.2f}, v28 {out['TAXITIME_SEC_mvt'].mean():.2f}")
res = validate_files(__import__("pathlib").Path(OUT), __import__("pathlib").Path(TPL))
print("gate:", "PASS" if res.ok else "FAIL", res.warnings)
res.raise_if_failed()
