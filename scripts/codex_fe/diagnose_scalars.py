"""Post-hoc decomposition of locked corrections; no new selection."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
import json
from pathlib import Path

import numpy as np
import polars as pl
from screen import correction, load

f = load()
tr = f.filter(pl.col("month").is_in([9, 10]))
te = f.filter(pl.col("month") >= 11)
base = te["pred"].to_numpy()
y = te["TAXITIME_SEC_mvt"].to_numpy()
apt = correction(tr, te, "airport_only")
selected = json.loads(Path("docs/codex_fe_results/selection.json").read_text())["selected"]
rows = []
for c in ["airport_only", *selected]:
    corr = correction(tr, te, c)
    p = np.maximum(0, base + corr)
    for month in [11, 12]:
        for regime in [0, 1]:
            m = (te["month"].to_numpy() == month) & (te["unmatched"].to_numpy() == regime)
            ds = (y - p) ** 2 - (y - base) ** 2
            rows.append(
                {
                    "candidate": c,
                    "month": month,
                    "unmatched": regime,
                    "n": int(m.sum()),
                    "null_fraction": None
                    if c == "airport_only"
                    else float(te.filter(pl.Series(m))[c].null_count() / m.sum()),
                    "correction_equal_airport_fraction": float(np.mean(corr[m] == apt[m])),
                    "delta_sse": float(ds[m].sum()),
                    "mean_correction": float(corr[m].mean()),
                    "mean_test_residual": float((y - base)[m].mean()),
                    "delta_rmse": float(np.sqrt(np.mean((y - p)[m] ** 2)) - np.sqrt(np.mean((y - base)[m] ** 2))),
                }
            )
Path("docs/codex_fe_results/scalar_diagnosis.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(r)
