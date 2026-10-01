"""OPDI airframe turnaround: how long this departure's aircraft had been on the ground (docs/opdi_2026-09-30/plan.md).

Source: EUROCONTROL's Open Performance Data Initiative flight list v0.0.2 (monthly parquet, built from OpenSky ADS-B),
https://www.opdi.aero. LICENCE FLAG: the files carry the EUROCONTROL PRU disclaimer (non-commercial, no modification
without permission), which may not meet the prize rule that every additional dataset be under an open-source licence
(docs/competitor_research_2026-09-21/results/16-external-data.md §6.1). Measured here; whether it may ship is the
team's decision, not this module's.

A departure is linked to its OPDI flight by callsign and the nearest `first_seen` (which is lift-off, median
+12 s from MVT_TIME) within 30 minutes, at any `adep`: half of OPDI's rows have no `adep`, and requiring it loses a
third of the LEMD and LTFM links. The airframe's previous OPDI flight then gives its landing time (`last_seen`).

Coverage is structured, not random: the January 2026 file has no flights at all on 2-10 and 23 January and a partial
24th, which is also the January snow crisis. Every other month of 2025-2026 is essentially complete.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import requests

from .config import data_dir

OPDI_URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{ym}.parquet"
OPDI_LICENCE = (
    "EUROCONTROL OPDI v0.0.2 (https://www.opdi.aero), EUROCONTROL PRU disclaimer - licence flag, see module doc"
)
MONTHS = [f"2025{m:02d}" for m in range(1, 13)] + ["202412", "202601", "202606", "202607"]
USER_AGENT = "prc-data-challenge-2026 feature research (resilient-hedgehog)"
LINK_TOL = "30m"
MAX_GROUND_S = 24 * 3600
# Under 15 minutes is 0.05% of links and is one flight's track split in two, not a turnaround (plan.md §1).
MIN_GROUND_S = 900
OPDI_COLS = ["opdi_ground_s", "opdi_sched_slack_s"]


def fetch_opdi(months: list[str] | None = None, cache_dir: Path | None = None) -> list[Path]:
    """Download the monthly flight lists that are not cached yet."""
    cache_dir = cache_dir or (data_dir() / "external" / "opdi")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ym in sorted(months or MONTHS):
        f = cache_dir / f"flight_list_{ym}.parquet"
        if not f.exists() or f.stat().st_size < 1_000_000:
            for attempt in range(4):
                r = requests.get(OPDI_URL.format(ym=ym), headers={"User-Agent": USER_AGENT}, timeout=900)
                if r.status_code == 200:
                    f.write_bytes(r.content)
                    break
                time.sleep(30 * (attempt + 1))
            else:
                raise RuntimeError(f"OPDI {ym}: HTTP {r.status_code}")
        out.append(f)
    return out


def _ts(col: str) -> pl.Expr:
    """The 2025 files type the timestamps as datetimes, the later ones as strings; normalise to naive UTC."""
    c = pl.col(col)
    return pl.coalesce(
        c.cast(pl.Utf8).str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        c.cast(pl.Utf8).str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False),
    ).alias(col)


def _scan(files: list[Path]) -> pl.LazyFrame:
    return pl.concat(
        [
            pl.scan_parquet(f).select(
                pl.col("flt_id").cast(pl.Utf8).str.strip_chars().alias("cs"),
                pl.col("icao24").cast(pl.Utf8),
                pl.col("ades").cast(pl.Utf8),
                _ts("first_seen"),
                _ts("last_seen"),
            )
            for f in files
        ],
        how="vertical",
    )


def turnaround_table(departures: pl.DataFrame, files: list[Path]) -> pl.DataFrame:
    """MVT_ID_mvt -> opdi_ground_s, opdi_sched_slack_s for every departure that links to an OPDI flight.

    `departures` needs MVT_ID_mvt, ADEP_mvt, CALLSIGN_flt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt. Reads no label.
    opdi_ground_s      = this flight's first_seen - the airframe's previous last_seen: time on the ground.
    opdi_sched_slack_s = SCHED_TIME - the previous last_seen: how much of the planned turnaround the inbound left.
    Both are null unless the previous flight ended at this departure's airport 15 minutes to 24 hours earlier."""
    dep = (
        departures.select(
            "MVT_ID_mvt",
            "ADEP_mvt",
            pl.col("CALLSIGN_flt").cast(pl.Utf8).str.strip_chars().alias("cs"),
            pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("t"),
            pl.col("SCHED_TIME_UTC_mvt").dt.replace_time_zone(None).alias("sched"),
        )
        .drop_nulls(["cs", "t"])
        .sort("t")
    )
    callsigns = dep["cs"].unique().implode()
    this = (
        _scan(files)
        .filter(pl.col("cs").is_in(callsigns) & pl.col("first_seen").is_not_null())
        .select(["cs", "icao24", "first_seen"])
        .collect()
        .sort("first_seen")
    )
    linked = dep.join_asof(
        this, left_on="t", right_on="first_seen", by="cs", strategy="nearest", tolerance=LINK_TOL
    ).drop_nulls("icao24")
    airframes = linked["icao24"].unique().implode()
    legs = (
        _scan(files)
        .filter(pl.col("icao24").is_in(airframes) & pl.col("first_seen").is_not_null())
        .select(["icao24", "first_seen", "last_seen", "ades"])
        .unique(subset=["icao24", "first_seen"], keep="first")
        .collect()
        .sort("first_seen")
        .rename({"first_seen": "prev_fs", "last_seen": "prev_ls", "ades": "prev_ades"})
    )
    prev = linked.sort("first_seen").join_asof(
        legs,
        left_on="first_seen",
        right_on="prev_fs",
        by="icao24",
        strategy="backward",
        allow_exact_matches=False,
    )
    ground = (pl.col("first_seen") - pl.col("prev_ls")).dt.total_seconds()
    ok = (pl.col("prev_ades") == pl.col("ADEP_mvt")) & (ground >= MIN_GROUND_S) & (ground <= MAX_GROUND_S)
    return prev.select(
        "MVT_ID_mvt",
        pl.when(ok).then(ground).cast(pl.Float64).alias("opdi_ground_s"),
        pl.when(ok)
        .then((pl.col("sched") - pl.col("prev_ls")).dt.total_seconds())
        .cast(pl.Float64)
        .alias("opdi_sched_slack_s"),
    ).unique(subset="MVT_ID_mvt", keep="first")


def build_turnaround_cache(raw_dir: Path | None = None, out: Path | None = None) -> Path:
    """The table for every training and ranking departure, cached once (it reads ~650 MB of OPDI)."""
    raw_dir = raw_dir or data_dir() / "raw"
    out = out or data_dir() / "work" / "opdi_turnaround.parquet"
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    files = sorted(raw_dir.glob("training_*.parquet")) + [raw_dir / "ranking.parquet"]
    deps = pl.concat(
        [pl.read_parquet(f, columns=cols).filter(pl.col("PHASE_mvt") == "DEP") for f in files], how="diagonal"
    )
    table = turnaround_table(deps, fetch_opdi())
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return out
