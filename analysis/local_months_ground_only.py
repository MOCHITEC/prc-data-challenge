"""Keep finding 2's swap (4-month residual B in place of Jan+Jul residual A) only on rows with ground ADS-B fixes;
air-only rows revert to the input file. 2025: on air-only rows A beat B in Jan (+99 vs +61 MSE) and Jul (+67 vs +24);
B's gain comes from ground rows (Nov+Dec +1,410 MSE overall).

    .venv/bin/python analysis/local_months_ground_only.py <before.parquet> <after_swap.parquet> <out.parquet>
"""
import os, sys
import polars as pl
a, b, dst = sys.argv[1:4]
A, B = pl.read_parquet(a), pl.read_parquet(b)
assert (A["MVT_ID_mvt"] == B["MVT_ID_mvt"]).all()
g = pl.read_parquet(os.path.join(os.environ["SCRATCH"], "learn", "feat_2026.parquet"), columns=["MVT_ID_mvt", "ground_fix"])
M = (A.rename({"TAXITIME_SEC_mvt": "a"}).join(B.rename({"TAXITIME_SEC_mvt": "b"}), on="MVT_ID_mvt", maintain_order="left")
     .join(g, on="MVT_ID_mvt", how="left", maintain_order="left"))
out = M.select("MVT_ID_mvt", pl.when(pl.col("ground_fix").fill_null(False)).then(pl.col("b")).otherwise(pl.col("a")).cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
assert (out["MVT_ID_mvt"] == A["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
out.write_parquet(dst)
d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - A["TAXITIME_SEC_mvt"].cast(pl.Float64)
print(f"{dst}: {int((d != 0).sum())} rows differ from the input, sum delta^2/N {float((d ** 2).sum()) / A.height:.0f}")
