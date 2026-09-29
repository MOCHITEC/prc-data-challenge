"""operator_aircraft_route 00: build the cached analysis frames.

Outputs (data/interim/, git-ignored):
  oar_dep.parquet  training + ranking departures, identity columns, clock-source label (training only)
  oar_arr.parquet  training + ranking arrivals (for ARR->DEP turnaround linking)
  data/external/ourairports_airports.csv  (public domain) for great-circle distance

Run:  POLARS_MAX_THREADS=2 .venv/bin/python analysis/operator_aircraft_route_00_prepare.py
"""

from __future__ import annotations

import glob
import os
import sys
import urllib.request

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prc.clocks import with_clock_source  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "interim")
EXT = os.path.join(ROOT, "data", "external")
os.makedirs(OUT, exist_ok=True)
os.makedirs(EXT, exist_ok=True)

DEP_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "TAXITIME_SEC_mvt", "LOBT_flt", "CALLSIGN_flt", "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt", "IOBT_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "AOBT_3_flt",
    "ARVT_1_flt", "ARVT_3_flt",
]
ARR_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "AOBT_3_flt", "ARVT_3_flt",
]


def load(phase: str, cols: list[str]) -> pl.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "training_*.parquet"))) + [os.path.join(RAW, "ranking.parquet")]:
        split = "rank" if "ranking" in f else "train"
        lf = pl.scan_parquet(f).filter(pl.col("PHASE_mvt") == phase).select(cols)
        frames.append(lf.collect().with_columns(pl.lit(split).alias("split")))
    df = pl.concat(frames, how="vertical_relaxed")
    return df.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("FLIGHT_ID_mvt").cast(pl.Int64))


def flight_prefix(col: str) -> pl.Expr:
    """Airline designator in FLIGHT_mvt: ICAO 3-letter when the string starts with 3 letters + digit, else the
    2-character IATA designator (may contain a digit, e.g. U2, W6)."""
    s = pl.col(col).str.to_uppercase().str.strip_chars()
    return (
        pl.when(s.str.contains(r"^[A-Z]{3}[0-9]"))
        .then(s.str.slice(0, 3))
        .when(s.str.contains(r"^[A-Z0-9]{2}[0-9]"))
        .then(s.str.slice(0, 2))
        .otherwise(None)
    )


def add_identity(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("CALLSIGN_flt").str.contains(r"^[A-Z]{3}"))
        .then(pl.col("CALLSIGN_flt").str.slice(0, 3))
        .otherwise(None)
        .alias("cs3"),
        flight_prefix("FLIGHT_mvt").alias("fl_pfx"),
    )


def main() -> None:
    dep = add_identity(load("DEP", DEP_COLS))
    arr = add_identity(load("ARR", ARR_COLS))
    both = pl.concat([dep.select("cs3", "fl_pfx"), arr.select("cs3", "fl_pfx")])
    # designator -> ICAO airline map learnt from NM-matched rows (no labels involved; train+rank both fine)
    m = (
        both.drop_nulls()
        .group_by("fl_pfx", "cs3")
        .len()
        .sort("len", descending=True)
        .group_by("fl_pfx", maintain_order=True)
        .agg(pl.col("cs3").first().alias("pfx_icao"), (pl.col("len").first() / pl.col("len").sum()).alias("pfx_purity"),
             pl.col("len").sum().alias("pfx_n"))
    )
    m.write_parquet(os.path.join(OUT, "oar_prefix_map.parquet"))

    def airline(df: pl.DataFrame) -> pl.DataFrame:
        return df.join(m.select("fl_pfx", "pfx_icao"), on="fl_pfx", how="left").with_columns(
            pl.coalesce("cs3", "pfx_icao", "fl_pfx").alias("airline")
        )

    dep = airline(dep)
    arr = airline(arr)
    lab = dep.filter(pl.col("split") == "train")
    lab = with_clock_source(lab, with_rollover=True)
    dep = pl.concat([lab, dep.filter(pl.col("split") == "rank").with_columns(pl.lit(None, pl.String).alias("clock_src"))])
    dep = dep.with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
    )
    dep.write_parquet(os.path.join(OUT, "oar_dep.parquet"))
    arr.write_parquet(os.path.join(OUT, "oar_arr.parquet"))
    print(dep.shape, arr.shape, m.shape)

    ap = os.path.join(EXT, "ourairports_airports.csv")
    if not os.path.exists(ap):
        try:
            urllib.request.urlretrieve("https://davidmegginson.github.io/ourairports-data/airports.csv", ap)
        except Exception as e:  # network optional; distance analysis falls back to ICAO prefix
            print("OurAirports download failed:", e)


if __name__ == "__main__":
    main()
