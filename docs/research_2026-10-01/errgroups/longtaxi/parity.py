import numpy as np, polars as pl
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet").with_columns(pl.col(pl.Float32).fill_nan(None))
TX=[c for c in tx.columns if c.startswith("tx_")]
# parity: NaN rate, 2025 Jan+Jul vs 2026
s=tx.with_columns(pl.col("lday").dt.month().alias("m")).filter(pl.col("m").is_in([1,7]))
nr=s.group_by("is_rank").agg([pl.col(c).null_count().truediv(pl.len()).round(4) for c in TX]).sort("is_rank")
print(nr.transpose(include_header=True).filter(pl.col("column_0")!=pl.col("column_1")))
# extrapolation: rows in 2026 beyond 2025 airport max
mx=tx.filter(~pl.col("is_rank")).group_by("ADEP_mvt").agg([pl.col(c).max().alias(c) for c in ["tx_mma_c60_ratio","tx_tir_c60","tx_day_tir","tx_day_mma_ratio","tx_day_late3h"]])
r=tx.filter("is_rank").join(mx,on="ADEP_mvt",suffix="_mx")
print(r.group_by("ADEP_mvt").agg([(pl.col(c)>pl.col(c+"_mx")).sum().alias(c) for c in ["tx_mma_c60_ratio","tx_tir_c60","tx_day_tir","tx_day_mma_ratio","tx_day_late3h"]]+[pl.len()]).sort("ADEP_mvt"))
