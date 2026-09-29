"""Shared helpers for the operator_aircraft_route analysis scripts."""

from __future__ import annotations

import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERIM = os.path.join(ROOT, "data", "interim")
WIDEBODY_PREFIX = ("A33", "A34", "A35", "A38", "B74", "B76", "B77", "B78", "IL9", "MD1", "A30", "A31")
# A31 = A310 (widebody); careful: A318/A319 are A31x narrowbodies -> resolved in is_widebody()


def load_dep(cols: list[str] | None = None) -> pl.DataFrame:
    p = os.path.join(INTERIM, "oar_dep.parquet")
    return pl.read_parquet(p, columns=cols) if cols else pl.read_parquet(p)


def load_arr(cols: list[str] | None = None) -> pl.DataFrame:
    p = os.path.join(INTERIM, "oar_arr.parquet")
    return pl.read_parquet(p, columns=cols) if cols else pl.read_parquet(p)


def is_widebody(col: str = "AIRCRAFT_TYPE_mvt") -> pl.Expr:
    t = pl.col(col)
    return (
        t.str.contains(r"^(A33|A34|A35|A38|B74|B76|B77|B78|IL96|MD11|A306|A30B|A310|B764|C5M|A124|A225|AN12)")
        .fill_null(False)
    )


def md(df: pl.DataFrame, floatfmt: str = "{:,.1f}") -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            elif isinstance(v, int) and not isinstance(v, bool):
                cells.append(f"{v:,}")
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def paired_boot(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """RMSE(b) - RMSE(a) with a 95% bootstrap CI (row resampling)."""
    rng = np.random.default_rng(seed)
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    N = len(y)
    d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(np.sqrt(eb[i].mean()) - np.sqrt(ea[i].mean()))
    d = np.array(d)
    return float(np.sqrt(eb.mean()) - np.sqrt(ea.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def flight_format() -> pl.Expr:
    """Fingerprint of how the airport's system wrote FLIGHT_mvt (a reporting-system signature, docs/analysis)."""
    F = pl.col("FLIGHT_mvt")
    garbled = (
        (F.str.len_chars() == 8)
        & F.str.contains("^[A-Z]{3}")
        & ((F.str.slice(1, 2) == F.str.slice(3, 2)) | (F.str.slice(2, 1) == F.str.slice(3, 1)))
    )
    return (
        pl.when(F.is_null()).then(pl.lit("null"))
        .when(F == pl.col("CALLSIGN_flt")).then(pl.lit("eq_callsign"))
        .when(garbled).then(pl.lit("garbled8"))
        .when(F.str.contains(r"^[A-Z]{3}[0-9]")).then(pl.lit("icao"))
        .when(F.str.contains(r"^[A-Z0-9]{2}[0-9]")).then(pl.lit("iata"))
        .otherwise(pl.lit("other"))
    )
