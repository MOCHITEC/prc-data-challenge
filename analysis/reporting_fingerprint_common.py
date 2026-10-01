"""Shared loader for the reporting_fingerprint analysis (docs/analysis/reporting_fingerprint.md).

Builds a compact cache of all training + ranking movements with the clock-source label (prc.clocks rules,
+-60s, priority aobt3 > sched > eobt1 > lobt > iobt, plus rollover_24h) for DEP rows of the training side.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = Path(os.environ.get("RF_CACHE", "/tmp/claude-0/-home-user-prc-data-challenge/5280a43e-f3f5-505a-9d0f-0d22dc433c63/scratchpad/rf_cache"))
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
    "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "CALLSIGN_flt", "MARKET_SEGMENT_flt", "IOBT_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
    "FLIGHT_RULE_mvt", "ADES_FILED_flt", "ADES_flt", "WK_TBL_CAT_flt",
]
CAND = {"aobt3": "AOBT_3_flt", "sched": "SCHED_TIME_UTC_mvt", "eobt1": "EOBT_1_flt", "lobt": "LOBT_flt", "iobt": "IOBT_flt"}
TOL = 60


def _read(path: Path, split: str) -> pl.DataFrame:
    return pl.scan_parquet(path).select(COLS).with_columns(pl.lit(split).alias("split")).collect()


def load_all(refresh: bool = False) -> pl.DataFrame:
    out = CACHE / "all.parquet"
    if out.exists() and not refresh:
        return pl.read_parquet(out)
    parts = [_read(p, "train") for p in sorted(RAW.glob("training_*.parquet"))]
    parts.append(_read(RAW / "ranking.parquet", "rank"))
    df = pl.concat(parts, how="vertical_relaxed")
    df = df.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    # clock source label (DEP rows with labels only)
    gap = lambda c: (pl.col("BLOCK_TIME_UTC_mvt") - pl.col(c)).dt.total_seconds().abs()  # noqa: E731
    chain = pl.when((pl.col("TAXITIME_SEC_mvt") - 86400).abs() < 5400).then(pl.lit("rollover_24h"))
    for k, c in CAND.items():
        chain = chain.when(gap(c) <= TOL).then(pl.lit(k))
    df = df.with_columns(
        pl.when((pl.col("PHASE_mvt") == "DEP") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
        .then(chain.otherwise(pl.lit("genuine"))).otherwise(None).alias("clock_src"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("date"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"),
        # airport = ADEP for departures, ADES for arrivals
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"),
    )
    df.write_parquet(out)
    return df


def sec(col: str) -> pl.Expr:
    """seconds-of-minute of a timestamp (integer part)."""
    return pl.col(col).dt.second()


def subsec(col: str) -> pl.Expr:
    return pl.col(col).dt.microsecond()
