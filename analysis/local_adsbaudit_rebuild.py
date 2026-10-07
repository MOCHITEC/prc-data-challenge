"""adsbaudit: rebuild per-day residual features with today's code and compare with the stored feature files
(2025 Jan+Jul files were built 2026-10-01, the 2026 ones 2026-10-05). Local, read-only; writes nothing.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsbaudit_rebuild.py 2025-01-05 2026-07-10 ...
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import adsb_learn_features as L  # noqa: E402

for day in sys.argv[1:]:
    new = L.day_features(day).sort("MVT_ID_mvt")
    old = pl.read_parquet(os.path.join(L.OUT, f"feat_{day}.parquet")).sort("MVT_ID_mvt")
    print(day, "rows new/old", new.height, old.height, "ids equal", new["MVT_ID_mvt"].equals(old["MVT_ID_mvt"]))
    if not new["MVT_ID_mvt"].equals(old["MVT_ID_mvt"]):
        new = new.join(old.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi").sort("MVT_ID_mvt")
        old = old.join(new.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi").sort("MVT_ID_mvt")
    bad = []
    for c in old.columns:
        if c not in new.columns:
            bad.append((c, "missing in new"))
            continue
        a, b = new[c], old[c]
        if a.dtype.is_numeric() and b.dtype.is_numeric():
            x, y = a.cast(pl.Float64).to_numpy(), b.cast(pl.Float64).to_numpy()
            diff = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-6))
            n = int(diff.sum())
        else:
            n = int((a.cast(pl.String).fill_null("<n>") != b.cast(pl.String).fill_null("<n>")).sum())
        if n:
            bad.append((c, n))
    print("  columns differing:", bad or "none", "| new-only columns:", [c for c in new.columns if c not in old.columns])
