"""Copied unchanged from claude/model-b-and-arrival-features (docs/competitor_research_2026-09-21/scripts/c12_plus_appear12.py, 9c881cc).

C12 (v42) + beautiful-davinci's 12-month appear delta (handoff/2026-10-06/appear12_delta.parquet, commit dd6e60d).

The table is a fixed per-row integer delta (v39 base, b recovered with v38's 4-month plain residual). It is added as
APPEAR12.md says, with three C12-specific exclusions (rows whose C12 value is set by an overlay, not by the residual
chain the delta corrects):
  - LIRF copy band (rule C of analysis/local_lirf_artefacts_apply.py: unmatched LIRF, 7 h <= (ms - 1100)/3600 <= 13.5 h)
  - LIRF time-zone class (rule B: unmatched LIRF, 0 <= ms < 1 h, prefix with q > 0)
  - rows sitting on the LOBT bound in C12 (C12's last step clipped them; their pre-clip value is unknown)
Then C12's LOBT clip is re-applied, and any negative value is reverted to C12. Nothing is uploaded.

    .venv/bin/python docs/competitor_research_2026-09-21/scripts/c12_plus_appear12.py <c12.parquet> <appear12_delta.parquet> <out.parquet>
"""
import glob
import sys

import numpy as np
import polars as pl

C12, DELTA, DST = sys.argv[1:4]
tpl = pl.read_parquet("data/raw/submitting.parquet")
sub = pl.read_parquet(C12)
assert sub.schema == tpl.schema and sub.height == tpl.height and (sub["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
d = pl.read_parquet(DELTA)
assert d["MVT_ID_mvt"].is_unique().all()


def unm(df):
    off = 3600 + pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.dst_offset().dt.total_seconds()
    return (df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
            .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                          pre=pl.col("FLIGHT_mvt").str.slice(0, 3).fill_null("?"), off=off))


# overlay rows of C10 (same rules and q table as local_lirf_artefacts_apply.py)
tr = unm(pl.concat([pl.scan_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))]).collect())
small = tr.filter(pl.col("ms").is_between(0, 3599)).with_columns(
    tz=((pl.col("TAXITIME_SEC_mvt") - pl.col("off")).is_between(240, 2700)) & ((pl.col("TAXITIME_SEC_mvt") - pl.col("ms")).abs() > 12))
qp = small.group_by("pre").agg(t=pl.col("tz").sum(), n=pl.len()).filter(pl.col("t") >= 1)
R = unm(pl.read_parquet("data/raw/ranking.parquet")).join(qp, on="pre", how="left").with_columns(
    dh=(pl.col("ms") - 1100) / 3600, qtz=(pl.col("t") / (pl.col("n") + 1)).fill_null(0))
ov_copy = R.filter(pl.col("dh").is_between(7, 13.5))["MVT_ID_mvt"]
ov_tz = R.filter(pl.col("ms").is_between(0, 3599) & (pl.col("qtz") > 0))["MVT_ID_mvt"]

f = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", "mvt_minus_lobt"])
G = (sub.rename({"TAXITIME_SEC_mvt": "c12"}).join(d, on="MVT_ID_mvt", how="left", maintain_order="left")
     .join(f, on="MVT_ID_mvt", how="left", maintain_order="left"))
lo, hi = (pl.col("mvt_minus_lobt") - 3606).ceil(), (pl.col("mvt_minus_lobt") + 3606).ceil()
G = G.with_columns(
    intab=pl.col("delta").is_not_null(),
    x_copy=pl.col("MVT_ID_mvt").is_in(ov_copy.implode()),
    x_tz=pl.col("MVT_ID_mvt").is_in(ov_tz.implode()),
    x_lobt=pl.col("mvt_minus_lobt").is_not_null() & ((pl.col("c12") == lo) | (pl.col("c12") == hi)))
G = G.with_columns(excl=pl.col("intab") & (pl.col("x_copy") | pl.col("x_tz") | pl.col("x_lobt")))
for c in ("x_copy", "x_tz", "x_lobt"):
    print(f"table rows excluded by {c}: {int((G['intab'] & G[c]).sum())}")
G = G.with_columns(raw=pl.col("c12") + pl.when(pl.col("excl")).then(0).otherwise(pl.col("delta").fill_null(0)))
clipped = pl.when(pl.col("mvt_minus_lobt").is_null()).then(pl.col("raw")).otherwise(pl.max_horizontal(pl.min_horizontal(pl.col("raw"), hi), lo))
G = G.with_columns(clipd=clipped)
n_clip = int((G["clipd"] != G["raw"]).sum())
n_neg = int((G["clipd"] < 0).sum())
G = G.with_columns(new=pl.when(pl.col("clipd") < 0).then(pl.col("c12")).otherwise(pl.col("clipd")).cast(pl.Int32))
print(f"table rows {int(G['intab'].sum()):,}; excluded {int(G['excl'].sum())}; re-clipped to LOBT bound {n_clip}; "
      f"negatives reverted to C12 {n_neg}")
out = G.select("MVT_ID_mvt", pl.col("new").alias("TAXITIME_SEC_mvt"))
assert out.schema == tpl.schema and out.height == tpl.height and (out["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
assert out.null_count().sum_horizontal().item() == 0 and int((out["TAXITIME_SEC_mvt"] < 0).sum()) == 0
out.write_parquet(DST)
dd = (out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)).to_numpy()
ch = dd != 0
print(f"wrote {DST}: rows changed {int(ch.sum()):,}; RMS over changed {np.sqrt(np.mean(dd[ch] ** 2)):.2f} s; "
      f"RMS over all {np.sqrt(np.mean(dd ** 2)):.2f} s; sum d^2/N {float((dd ** 2).sum()) / len(dd):.0f}; "
      f"max |d| {np.abs(dd).max():.0f}; mean {dd.mean():+.3f}")
