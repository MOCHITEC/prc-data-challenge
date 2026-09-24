"""How much RMSE is in *level* errors at all? Oracle de-biasing of the proxy OOF (clean rows) per airport-month,
per airport-day and per airport-day-hour. Upper bound on any level-shift correction (ARR taxi-in proxy, month
weighting, day-level features). Output: docs/analysis/temporal_shift.md §2."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from temporal_shift_common import CACHE  # noqa: E402
from temporal_shift_features import features  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
f = features().filter(pl.col("set") == "train").select("MVT_ID_mvt", "APT", "date_local", "hour_local", "y")
for m in (1, 7):
    o = pl.read_parquet(CACHE / f"oofc_{tag}_m{m}.parquet").join(f, on="MVT_ID_mvt")
    o = o.filter(pl.col("y").is_between(30, 7200)).with_columns((pl.col("y") - pl.col("pred")).alias("r"))
    base = float(np.sqrt((o["r"] ** 2).mean()))
    out = {"fold": m, "rmse_clean": round(base, 2)}
    for name, keys in (("apt", ["APT"]), ("apt_day", ["APT", "date_local"]), ("apt_day_hour", ["APT", "date_local", "hour_local"])):
        r2 = o.with_columns((pl.col("r") - pl.col("r").mean().over(keys)).alias("r2"))["r2"]
        out[f"oracle_{name}"] = round(float(np.sqrt((r2 ** 2).mean())), 2)
    print(out)
