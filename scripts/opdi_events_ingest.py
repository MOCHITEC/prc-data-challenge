"""Download OPDI v002 flight events and keep the ground events at the ten challenge airports (docs/opdi_events_2026-10-05/).

    .venv/bin/python scripts/opdi_events_ingest.py      # writes data/external/opdi_events/ground_<span>.parquet

LICENCE FLAG: OPDI is published by EUROCONTROL under its general disclaimer (attribution, non-commercial, no
modification without permission). The challenge's prize rule asks for additional datasets "openly available under an
open source license"; whether OPDI qualifies is the user's call. Leaderboard use is not affected.

Each 10-day file is fetched, filtered to the airport-tagged ground events (stand, apron, taxiway, runway, de-icing pad)
at the ten airports plus the take-off / first_seen events of the same flights, written compactly, and the raw file is
deleted. Labels are never read here."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import requests

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{span}.parquet"
OUT = Path("data/external/opdi_events")
UA = "prc-data-challenge-2026 research (resilient-hedgehog)"
SPANS = [
    "20241226_20250105",
    "20250105_20250115",
    "20250115_20250125",
    "20250125_20250204",
    "20250204_20250214",
    "20250214_20250224",
    "20250224_20250306",
    "20250306_20250316",
    "20250316_20250326",
    "20250326_20250405",
    "20250405_20250415",
    "20250415_20250425",
    "20250425_20250505",
    "20250505_20250515",
    "20250515_20250525",
    "20250525_20250604",
    "20250604_20250614",
    "20250614_20250624",
    "20250624_20250704",
    "20250704_20250714",
    "20250714_20250724",
    "20250724_20250803",
    "20250803_20250813",
    "20250813_20250823",
    "20250823_20250902",
    "20250902_20250912",
    "20250912_20250922",
    "20250922_20251002",
    "20251002_20251012",
    "20251012_20251022",
    "20251022_20251101",
    "20251101_20251111",
    "20251111_20251121",
    "20251121_20251201",
    "20251201_20251211",
    "20251211_20251221",
    "20251221_20251231",
    "20251231_20260110",
    "20260110_20260120",
    "20260120_20260130",
    "20260130_20260209",
    "20260629_20260709",
    "20260709_20260719",
    "20260719_20260729",
    "20260729_20260808",
]


def filter_file(raw: Path) -> pl.DataFrame:
    d = pl.read_parquet(raw, columns=["flight_id", "type", "event_time", "info"])
    # 2025 files store event_time as a datetime, 2026 files as "YYYY-MM-DD HH:MM:SS" text
    if d.schema["event_time"] == pl.String:
        d = d.with_columns(pl.col("event_time").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
    d = d.with_columns(pl.col("event_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    j = pl.col("info")
    ground = (
        d.filter(j.str.contains('"osm_airport"'))
        .with_columns(
            j.str.json_path_match("$.osm_airport").alias("airport"),
            j.str.json_path_match("$.osm_aeroway").alias("aeroway"),
            j.str.json_path_match("$.osm_ref").alias("ref"),
            j.str.json_path_match("$.osn_flight_id").str.strip_chars().alias("callsign"),
            j.str.json_path_match("$.osn_icao24").alias("icao24"),
            j.str.json_path_match("$.opdi_time_in_use_s").cast(pl.Float64, strict=False).alias("in_use_s"),
        )
        .filter(pl.col("airport").is_in(AIRPORTS))
    )
    ids = ground["flight_id"].unique()
    other = d.filter(
        pl.col("type").is_in(["take-off", "first_seen", "landing", "last_seen"])
        & pl.col("flight_id").is_in(ids.implode())
    )
    keep = ["flight_id", "type", "event_time", "airport", "aeroway", "ref", "callsign", "icao24", "in_use_s"]
    other = other.with_columns(
        [pl.lit(None, dtype=pl.String).alias(c) for c in ("airport", "aeroway", "ref", "callsign", "icao24")]
        + [pl.lit(None, dtype=pl.Float64).alias("in_use_s")]
    )
    out = pl.concat([ground.select(keep), other.select(keep)])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for span in SPANS:
        dst = OUT / f"ground_{span}.parquet"
        if dst.exists():
            continue
        raw = OUT / f"flight_events_{span}.parquet"
        if not raw.exists():
            for attempt in range(4):
                try:
                    with requests.get(URL.format(span=span), headers={"User-Agent": UA}, timeout=900, stream=True) as r:
                        r.raise_for_status()
                        tmp = raw.with_suffix(".part")
                        with open(tmp, "wb") as fh:
                            fh.writelines(r.iter_content(1 << 20))
                        tmp.rename(raw)
                    break
                except requests.RequestException as exc:
                    print(f"{span}: attempt {attempt + 1} failed: {exc!r}", flush=True)
                    time.sleep(2 ** (attempt + 1))
            else:
                print(f"{span}: FAILED", flush=True)
                continue
        out = filter_file(raw)
        out.write_parquet(dst)
        n_exit = out.filter(pl.col("type") == "exit-parking_position").height
        print(
            f"{span}: raw {raw.stat().st_size / 1e6:.0f} MB -> {out.height:,} rows, exit-parking {n_exit:,}", flush=True
        )
        raw.unlink()


if __name__ == "__main__":
    main()
