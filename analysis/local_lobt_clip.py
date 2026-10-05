"""Clip predictions into the LOBT identity: |y - (MVT - LOBT)| <= 3,606 s holds on every one of the 2,062,623 matched
2025 training rows (min -3606, max +3606). Rows outside it in a submission can only get closer to the truth.

    .venv/bin/python analysis/local_lobt_clip.py <in.parquet> <out.parquet>
"""
import sys
import polars as pl
src, dst = sys.argv[1:3]
sub = pl.read_parquet(src)
f = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", "mvt_minus_lobt"])
D = sub.join(f, on="MVT_ID_mvt", how="left", maintain_order="left")
lo, hi = pl.col("mvt_minus_lobt") - 3606, pl.col("mvt_minus_lobt") + 3606
out = D.with_columns(pl.when(pl.col("mvt_minus_lobt").is_null()).then(pl.col("TAXITIME_SEC_mvt"))
                     .otherwise(pl.max_horizontal(pl.min_horizontal(pl.col("TAXITIME_SEC_mvt"), hi.ceil()), lo.ceil()))
                     .cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns)
assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
out.write_parquet(dst)
print(f"{dst}: {int((d != 0).sum())} rows clipped, sum delta^2/N {float((d ** 2).sum()) / sub.height:.2f}")
