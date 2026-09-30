"""Shared loaders for the stand / spatial-structure analysis (docs/analysis/stand.md).

Everything here uses only columns that survive in ranking.parquet for departures (MVT_TIME, STAND, RUNWAY,
NM clocks, SCHED) plus arrival rows (whose BLOCK_TIME / TAXITIME are present in ranking). Departure labels are
loaded only as the target / for diagnostics, never as inputs to stand-event features.
"""

from __future__ import annotations

import glob
import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
SCRATCH = os.environ.get("STAND_SCRATCH", "/tmp/stand_analysis")
os.makedirs(SCRATCH, exist_ok=True)

BAD_STANDS = {"UNKNOWN", "NA", "", "NULL", "0"}


def load_all(extra: tuple[str, ...] = (), keep_stand_str: bool = False) -> pl.DataFrame:
    """Training (split='train') + ranking (split='rank') movements with a unified airport/stand key.

    Kept lean for the 2.5 GB budget: timestamps become float epoch seconds, the (airport, stand) key becomes an
    integer SKEY, and only `extra` raw columns are carried along."""
    base = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
    cols = base + [c for c in extra if c not in base]
    frames = []
    files = sorted(glob.glob(os.path.join(RAW, "training_*.parquet"))) + [os.path.join(RAW, "ranking.parquet")]
    for f in files:
        split = "rank" if f.endswith("ranking.parquet") else "train"
        lf = pl.scan_parquet(f).select(cols).with_columns(pl.lit(split).cast(pl.Categorical).alias("split"))
        frames.append(lf)
    lf = pl.concat(frames)
    ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)  # noqa: E731
    stand_ok = (
        pl.when(pl.col("STAND_mvt").is_null() | pl.col("STAND_mvt").str.to_uppercase().is_in(list(BAD_STANDS)))
        .then(None)
        .otherwise(pl.col("STAND_mvt"))
    )
    apt = pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt"))
    lf = lf.with_columns(
        apt.cast(pl.Categorical).alias("APT"),
        pl.col("PHASE_mvt").cast(pl.Categorical),
        ep("MVT_TIME_UTC_mvt").alias("t_mvt"),
        ep("BLOCK_TIME_UTC_mvt").alias("t_blk"),
        ep("AOBT_3_flt").alias("t_aobt"),
        ep("SCHED_TIME_UTC_mvt").alias("t_sched"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        (apt + "|" + stand_ok).alias("SKEY_str"),
        stand_ok.alias("STAND_ok"),
        pl.col("RUNWAY_mvt").cast(pl.Categorical),
    )
    drop = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
            "ADEP_mvt", "ADES_mvt", "STAND_mvt"]
    df = lf.drop([c for c in drop if c not in extra]).collect()
    df = df.with_columns(pl.col("SKEY_str").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("SKEY"))
    if not keep_stand_str:
        df = df.drop(["SKEY_str", "STAND_ok"])
    return df


def stand_prefix(s: pl.Expr) -> pl.Expr:
    """Leading letters of a stand id ('' for all-numeric). E.g. 'A14'->'A', 'V108'->'V', '221L'->''."""
    return s.str.extract(r"^([A-Za-z]*)", 1).fill_null("")


def stand_number(s: pl.Expr) -> pl.Expr:
    return s.str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)
