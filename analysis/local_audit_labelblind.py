"""Audit worker, step 1b: do any features read departure labels (BLOCK_TIME / TAXITIME of departures)?

Rebuilds the Jan+Jul 2025 feature table from raw files in which every departure's BLOCK_TIME_UTC_mvt and
TAXITIME_SEC_mvt are nulled BEFORE the build - byte-for-byte the ranking.parquet situation - and compares with
the 2-month rebuild that had the labels present (feat_jj2m.parquet from local_audit_frame.py). Identical => the
feature code never reads a departure label (its own or another flight's), so training and ranking rows are
treated the same way.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_audit_labelblind.py
Output: $SCRATCH/agents/audit/labelblind.txt
"""
import glob
import os
import sys

import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from local_audit_frame import visible  # noqa: E402
from prc.features import CATEGORICALS, build_departure_features  # noqa: E402
from prc.weather import join_weather  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "audit")
num = visible()
raw = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_2025-0[17]-01_*.parquet"))],
                how="diagonal_relaxed")
dep = pl.col("PHASE_mvt") == "DEP"
raw = raw.with_columns(pl.when(dep).then(None).otherwise(pl.col(c)).alias(c)
                       for c in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"))
f = join_weather(build_departure_features(raw, proxy_mode=None), pl.read_parquet("data/ref/metar_hourly.parquet"))
f = f.select(["MVT_ID_mvt", *num, *CATEGORICALS])
g = pl.read_parquet(os.path.join(OUT, "feat_jj2m.parquet"))
j = f.join(g, on="MVT_ID_mvt", suffix="__lab")
bad = []
for c in num + CATEGORICALS:
    a, b = j[c], j[c + "__lab"]
    if a.dtype.is_numeric():
        a, b = a.cast(pl.Float64).fill_nan(None), b.cast(pl.Float64).fill_nan(None)
        same = ((a - b).abs() <= 1e-6 * (1 + a.abs())) | (a.is_null() & b.is_null())
    else:
        same = (a == b) | (a.is_null() & b.is_null())
    n = int((~same.fill_null(False)).sum())
    if n:
        bad.append((c, n))
msg = f"rows {j.height:,} of {f.height:,}; columns differing when departure labels are blanked: {bad or 'none'}"
print(msg)
open(os.path.join(OUT, "labelblind.txt"), "w").write(msg + "\n")
