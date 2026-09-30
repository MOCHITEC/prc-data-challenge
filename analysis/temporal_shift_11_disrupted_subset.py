"""Proxy OOF on disrupted vs normal days: where do day-level features help? Output: temporal_shift.md §4."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from temporal_shift_common import CACHE  # noqa: E402
from temporal_shift_features import features  # noqa: E402

alt = sys.argv[1] if len(sys.argv) > 1 else "day"
f = features().filter(pl.col("set") == "train").with_columns(pl.col("APT").cast(pl.Utf8))
norm = f.group_by("APT").agg(pl.col("day_m_aobt3_mean").median().alias("norm"))
f = f.join(norm, on="APT").with_columns((pl.col("day_m_aobt3_mean") / pl.col("norm")).alias("ma_ratio"))
f = f.with_columns(pl.when((pl.col("day_msched_gt3h") > 0.02) | (pl.col("ma_ratio") > 1.3)).then(pl.lit("disrupted"))
                   .otherwise(pl.lit("normal")).alias("daytype")).select("MVT_ID_mvt", "APT", "y", "daytype", "m_aobt3")
for m in (1, 7):
    a = pl.read_parquet(CACHE / f"oofc_base_m{m}.parquet").rename({"pred": "p0"})
    b = pl.read_parquet(CACHE / f"oofc_{alt}_m{m}.parquet").rename({"pred": "p1"})
    o = a.join(b, on="MVT_ID_mvt").join(f, on="MVT_ID_mvt").filter(pl.col("y").is_between(30, 7200))
    t = o.group_by("daytype").agg(pl.len().alias("n"), ((pl.col("y") - pl.col("p0")) ** 2).mean().sqrt().alias("rmse_base"),
                                  ((pl.col("y") - pl.col("p1")) ** 2).mean().sqrt().alias(f"rmse_{alt}"),
                                  (pl.col("p0") - pl.col("y")).mean().alias("bias_base"),
                                  (pl.col("p1") - pl.col("y")).mean().alias(f"bias_{alt}"),
                                  pl.col("APT").unique().sort().str.join(",").alias("apts"))
    print(f"fold {m}")
    print(t)
    big = o.filter(pl.col("m_aobt3") > 2400)
    print("rows with MVT-AOBT3 > 2400 s:", big.height,
          "rmse base", float(np.sqrt(((big["y"] - big["p0"]) ** 2).mean())), f"rmse {alt}", float(np.sqrt(((big["y"] - big["p1"]) ** 2).mean())))
