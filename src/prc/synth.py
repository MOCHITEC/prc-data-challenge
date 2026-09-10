"""Synthetic movement data following the official schema (docs/01 §2.2).

Purpose: run the whole pipeline (features -> CV -> submission gate) before the real files arrive.
The generator imitates the documented structure (ARR+DEP rows, NM left join with unmatched rows,
airport/stand/runway effects, congestion, a heavy tail with 'reporting failure' regimes at one airport,
the ranking set with DEP block/taxi blanked). It is NOT a statistical model of the real data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .config import TARGET_AIRPORTS
from .schema import ALL_COLS, BLANKED_DEP_COLS

AIRPORT_BASE_SEC = {  # rough medians, seconds (CODA summer 2025 order of magnitude)
    "EDDF": 780,
    "EDDM": 780,
    "EGLL": 1140,
    "EHAM": 780,
    "LEBL": 1080,
    "LEMD": 1080,
    "LFPG": 960,
    "LIRF": 960,
    "LTFM": 1020,
    "LSZH": 780,
}
AC_TYPES = [
    "A320",
    "A21N",
    "A319",
    "B738",
    "B38M",
    "A333",
    "B77W",
    "A388",
    "E190",
    "CRJ9",
    "AT76",
    "B789",
]
WTC = {
    "A320": "M",
    "A21N": "M",
    "A319": "M",
    "B738": "M",
    "B38M": "M",
    "A333": "H",
    "B77W": "H",
    "A388": "J",
    "E190": "M",
    "CRJ9": "M",
    "AT76": "M",
    "B789": "H",
}
SEGMENTS = [
    "Mainline",
    "Regional",
    "Low-Cost",
    "Business Aviation",
    "All-Cargo",
    "Charter",
    "Other",
    "Not classified",
]
DESTS = [
    "EGKK",
    "LFPO",
    "EDDH",
    "LEPA",
    "LIML",
    "LOWW",
    "EBBR",
    "EKCH",
    "ESSA",
    "LPPT",
    "KJFK",
    "OMDB",
    "VHHH",
    "GCLP",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_month(
    year: int, month: int, seed: int, deps_per_airport_day: int = 120, blank_dep: bool = False
) -> pl.DataFrame:
    """Return one month of ARR+DEP movements for all target airports."""
    rng = _rng(seed)
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), (month % 12) + 1, 1)
    n_days = (end - start).days
    rows = []
    mvt_id = year * 10_000_000 + month * 100_000
    flight_id = mvt_id * 3
    for apt in TARGET_AIRPORTS:
        n_stands, n_rwy = 60, 3
        stands = [f"{chr(65 + i // 20)}{(i % 20) + 1:02d}" for i in range(n_stands)]
        runways = (
            [f"RW{k}" for k in ["25C", "18", "07C"][:n_rwy]]
            if apt == "EDDF"
            else [f"RW{k}" for k in ["27L", "27R", "09R"][:n_rwy]]
        )
        stand_eff = rng.normal(0, 150, n_stands)
        rwy_eff = rng.normal(0, 120, n_rwy)
        for d in range(n_days):
            day = start + timedelta(days=d)
            # hourly demand profile
            hours = rng.choice(24, size=deps_per_airport_day, p=_hour_profile())
            minutes = rng.integers(0, 60, size=deps_per_airport_day)
            for ph in ("DEP", "ARR"):
                for h, m in zip(hours, minutes):
                    sched = day + timedelta(hours=int(h), minutes=int(m))
                    matched = rng.random() > (0.06 if apt == "LIRF" else 0.015)
                    ac = rng.choice(AC_TYPES)
                    si, ri = rng.integers(n_stands), rng.integers(n_rwy)
                    congestion = 1.0 + 0.6 * _hour_profile()[h] * 24  # ~1..2.4
                    if ph == "DEP":
                        delay = rng.gamma(2.0, 300) - 300 + (600 if rng.random() < 0.05 else 0)
                        block = sched + timedelta(seconds=float(delay))
                        taxi = AIRPORT_BASE_SEC[apt] * congestion * rng.lognormal(0, 0.25) + stand_eff[si] + rwy_eff[ri]
                        taxi = max(120.0, taxi)
                        # reporting-failure regime: block reported as schedule (mostly unmatched LIRF rows)
                        if not matched and apt == "LIRF" and rng.random() < 0.08:
                            block = sched
                            if rng.random() < 0.2:
                                block = sched - timedelta(days=1)
                        mvt = (
                            block + timedelta(seconds=float(taxi))
                            if not (block == sched or block == sched - timedelta(days=1))
                            else sched + timedelta(seconds=float(delay + taxi))
                        )
                        taxi_sec = (mvt - block).total_seconds()
                        aobt3 = block + timedelta(seconds=float(rng.normal(0, 90))) if matched else None
                        adep, ades = apt, rng.choice(DESTS)
                    else:
                        landing = sched + timedelta(seconds=float(rng.normal(0, 600)))
                        taxi = AIRPORT_BASE_SEC[apt] * 0.5 * congestion * rng.lognormal(0, 0.25) + stand_eff[si] * 0.5
                        taxi = max(90.0, taxi)
                        mvt = landing
                        block = landing + timedelta(seconds=float(taxi))
                        taxi_sec = taxi
                        aobt3 = (
                            (sched - timedelta(hours=2) + timedelta(seconds=float(rng.normal(0, 600))))
                            if matched
                            else None
                        )
                        adep, ades = rng.choice(DESTS), apt
                    mvt_id += 1
                    flight_id += 1
                    eobt1 = (sched if ph == "DEP" else sched - timedelta(hours=2)) + timedelta(
                        seconds=float(rng.normal(0, 120))
                    )
                    rows.append(
                        {
                            "MVT_ID_mvt": mvt_id,
                            "FLIGHT_ID_mvt": flight_id if matched else None,
                            "FLIGHT_mvt": f"XX{rng.integers(100, 9999)}",
                            "FLIGHT_RULE_mvt": "I",
                            "ADEP_mvt": adep,
                            "ADES_mvt": ades,
                            "PHASE_mvt": ph,
                            "MVT_TIME_UTC_mvt": mvt,
                            "BLOCK_TIME_UTC_mvt": None if (blank_dep and ph == "DEP") else block,
                            "SCHED_TIME_UTC_mvt": sched,
                            "AIRCRAFT_TYPE_mvt": ac,
                            "RUNWAY_mvt": runways[ri],
                            "STAND_mvt": stands[si],
                            "TAXITIME_SEC_mvt": None if (blank_dep and ph == "DEP") else float(round(taxi_sec)),
                            "LOBT_flt": (eobt1 + timedelta(seconds=float(rng.normal(300, 300)))) if matched else None,
                            "CALLSIGN_flt": f"OP{rng.integers(1, 60):02d}{rng.integers(10, 999)}" if matched else None,
                            "ADEP_flt": adep if matched else None,
                            "ADES_flt": ades if matched else None,
                            "ADES_FILED_flt": ades if matched else None,
                            "MARKET_SEGMENT_flt": rng.choice(SEGMENTS) if matched else None,
                            "IOBT_flt": eobt1 if matched else None,
                            "FLIGHT_RULE_flt": "I" if matched else None,
                            "FLIGHT_TYPE_flt": "S" if matched else None,
                            "AIRCRAFT_TYPE_flt": ac if matched else None,
                            "WK_TBL_CAT_flt": WTC[ac] if matched else None,
                            "AIRCRAFT_OPERATOR_flt": f"OP{rng.integers(1, 60):02d}" if matched else None,
                            "EOBT_1_flt": eobt1 if matched else None,
                            "ARVT_1_flt": (eobt1 + timedelta(hours=2)) if matched else None,
                            "AOBT_3_flt": aobt3,
                            "ARVT_3_flt": (aobt3 + timedelta(hours=2)) if (matched and aobt3 is not None) else None,
                        }
                    )
    df = pl.DataFrame(
        rows,
        # Match the official parquet schema exactly (measured 2026-09-10 on the real files): ids are float64
        # holding integral values, the target is int32, and every timestamp carries a UTC zone. A synthetic
        # frame that differs here lets dtype bugs through to the first real run.
        schema_overrides={
            **{
                c: pl.Datetime("us", "UTC")
                for c in (
                    "MVT_TIME_UTC_mvt",
                    "BLOCK_TIME_UTC_mvt",
                    "SCHED_TIME_UTC_mvt",
                    "LOBT_flt",
                    "IOBT_flt",
                    "EOBT_1_flt",
                    "ARVT_1_flt",
                    "AOBT_3_flt",
                    "ARVT_3_flt",
                )
            },
            "MVT_ID_mvt": pl.Float64,
            "FLIGHT_ID_mvt": pl.Float64,
            "TAXITIME_SEC_mvt": pl.Int32,
        },
    )
    return df.select(ALL_COLS).sort("MVT_TIME_UTC_mvt")


def _hour_profile() -> np.ndarray:
    p = np.array(
        [
            0.3,
            0.2,
            0.1,
            0.1,
            0.3,
            1.2,
            2.5,
            3.0,
            2.6,
            2.2,
            2.0,
            2.2,
            2.4,
            2.3,
            2.2,
            2.4,
            2.6,
            2.8,
            2.6,
            2.2,
            1.8,
            1.2,
            0.7,
            0.4,
        ]
    )
    return p / p.sum()


def write_synthetic_dataset(
    out_dir: Path,
    months: list[tuple[int, int]] | None = None,
    seed: int = 7,
    deps_per_airport_day: int = 60,
) -> dict[str, Path]:
    """Write training_YYYY-MM-01_..parquet for the given 2025 months, plus ranking.parquet (Jan+Jul 2026 with DEP blanked)
    and submitting.parquet. Small by default so tests stay fast."""
    out_dir.mkdir(parents=True, exist_ok=True)
    months = months or [(2025, m) for m in (1, 2, 6, 7, 11, 12)]
    paths: dict[str, Path] = {}
    for i, (y, m) in enumerate(months):
        nxt = (y + (m == 12), (m % 12) + 1)
        name = f"training_{y}-{m:02d}-01_{nxt[0]}-{nxt[1]:02d}-01.parquet"
        generate_month(y, m, seed + i, deps_per_airport_day).write_parquet(out_dir / name)
        paths[name] = out_dir / name
    rank = pl.concat(
        [
            generate_month(2026, 1, seed + 101, deps_per_airport_day, blank_dep=True),
            generate_month(2026, 7, seed + 107, deps_per_airport_day, blank_dep=True),
        ]
    )
    rank.write_parquet(out_dir / "ranking.parquet")
    paths["ranking.parquet"] = out_dir / "ranking.parquet"
    sub = rank.filter(pl.col("PHASE_mvt") == "DEP").select(["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    sub.write_parquet(out_dir / "submitting.parquet")
    paths["submitting.parquet"] = out_dir / "submitting.parquet"
    return paths


def blanked_columns() -> list[str]:
    return list(BLANKED_DEP_COLS)
