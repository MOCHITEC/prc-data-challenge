"""Shared loader for the temporal_shift analysis (docs/analysis/temporal_shift.md).

Builds one compact parquet with every movement (training 2025 + ranking 2026) and the columns the analysis needs,
plus derived columns that are observable at ranking time (clock gaps, local time) and, for labelled departures,
the clock-source label (same definition as src/prc/clocks.py, tolerance 60 s).

Run:  POLARS_MAX_THREADS=2 .venv/bin/python analysis/temporal_shift_common.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc.clocks import with_clock_source  # noqa: E402
from prc.features import AIRPORT_TZ  # noqa: E402

RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "interim" / "temporal_shift"
BASE = CACHE / "base.parquet"

COLS = [
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt",
    "EOBT_1_flt",
    "LOBT_flt",
    "IOBT_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "CALLSIGN_flt",
    "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt",
    "FLIGHT_RULE_mvt",
    "TAXITIME_SEC_mvt",
]


def _sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def build_base() -> pl.DataFrame:
    files = sorted(RAW.glob("training_2025-*.parquet")) + [RAW / "ranking.parquet"]
    frames = []
    for f in files:
        lf = pl.scan_parquet(f).select(COLS)
        frames.append(lf.with_columns(pl.lit("rank" if f.name.startswith("ranking") else "train").alias("set")))
    df = pl.concat(frames, how="vertical_relaxed").collect()
    ts = df.schema["MVT_TIME_UTC_mvt"]
    df = df.with_columns([pl.col(c).cast(ts) for c in ("SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt",
                                                       "LOBT_flt", "IOBT_flt", "BLOCK_TIME_UTC_mvt")])
    df = df.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("APT"),
        _sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("m_aobt3"),
        _sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("m_sched"),
        _sec("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("m_eobt1"),
        _sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("m_lobt"),
        _sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt3_m_sched"),
        pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8).alias("nm"),
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        pl.col("CALLSIGN_flt").str.slice(0, 3).alias("airline"),
    )
    # airport-local clock (the airport of the movement)
    parts = []
    for apt, tz in AIRPORT_TZ.items():
        p = df.filter(pl.col("APT") == apt)
        loc = p["MVT_TIME_UTC_mvt"].dt.convert_time_zone(tz)
        parts.append(p.with_columns(loc.dt.hour().alias("hour_local"), loc.dt.date().alias("date_local"),
                                    loc.dt.weekday().alias("dow_local")))
    df = pl.concat(parts)
    # clock-source label on labelled departures only
    lab = df.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    lab = with_clock_source(lab.select("MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
                                       "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"))
    df = df.join(lab.select("MVT_ID_mvt", "clock_src"), on="MVT_ID_mvt", how="left")
    for c in ("PHASE_mvt", "ADEP_mvt", "APT", "RUNWAY_mvt", "set", "clock_src", "MARKET_SEGMENT_flt",
              "WK_TBL_CAT_flt", "FLIGHT_RULE_mvt"):
        df = df.with_columns(pl.col(c).cast(pl.Categorical))
    return df


def load(columns: list[str] | None = None, refresh: bool = False) -> pl.DataFrame:
    if refresh or not BASE.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        build_base().write_parquet(BASE)
    lf = pl.scan_parquet(BASE)
    return (lf.select(columns) if columns else lf).collect()


if __name__ == "__main__":
    d = load(refresh=True)
    print(d.shape, d.estimated_size("mb"))
    print(d.group_by("set", "PHASE_mvt").len())
