"""plan_standmap12.md section 1: the own-stand tables fill12 and rand12, and the appear columns rebuilt on them.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/stand_map12.py

fill12 = fill (OSM, else the 2026 map), else a map from the stand runs of 2025 and 2026 under the same rule as
stand_map.build_map (stand_dur >= 600 s, last fix of the run, callsign / flight-number join, median per (airport,
normalised stand), >= 5 departures, spread <= 40 m). A 2025 row gets the map without its own month's runs (12 maps), so no
row's own stand run places its own stand; a 2026 row gets the map of all runs. rand12 = fill12 with the coordinates of the
stands that come from that map permuted among themselves within each airport (same own_ok, wrong positions).
Writes stand_runs_12.parquet, appear_{fill12,rand12}_<day>.parquet and appear_{fill12,rand12}_<key>.parquet."""

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_appear as A  # noqa: E402
import adsb_learn_features as L  # noqa: E402
import adsb_stage1_feasibility as s1  # noqa: E402
import stand_map as SM  # noqa: E402
from build_features import DAYS  # noqa: E402

from prc.osm import normalise_stand  # noqa: E402

KEYS = ("2025", "2025nd", "2025m8", "2026")
RUNS = os.path.join(L.OUT, "stand_runs_12.parquet")


def month_of(day: str) -> str:
    return day[:7] if day.startswith("2025") else "2026"


def runs() -> pl.DataFrame:
    if os.path.exists(RUNS):
        return pl.read_parquet(RUNS)
    rows = []
    for key in KEYS:
        for d in DAYS[key]:
            jp = os.path.join(s1.SCRATCH, f"{L.tag_of(d)}_joined.parquet")
            if not os.path.exists(jp):
                continue
            J = pl.read_parquet(jp, columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "icao", "join", "t_move_lo", "stand_dur"])
            J = J.filter(pl.col("icao").is_not_null() & pl.col("join").is_in(["cs", "fn"]) & (pl.col("stand_dur") >= SM.MIN_DUR)
                         & pl.col("t_move_lo").is_not_null())
            fx = pl.read_parquet(os.path.join(s1.SCRATCH, f"{L.tag_of(d)}.parquet"), columns=["icao", "apt", "t", "lat", "lon"])
            P = J.join(fx.rename({"apt": "ADEP_mvt", "t": "t_move_lo"}), on=["icao", "ADEP_mvt", "t_move_lo"], how="inner")
            rows.append(P.unique("MVT_ID_mvt").select("ADEP_mvt", "STAND_mvt", "lat", "lon", pl.lit(month_of(d)).alias("month")))
    P = pl.concat(rows).with_columns(
        pl.col("STAND_mvt").cast(pl.Utf8).map_elements(normalise_stand, return_dtype=pl.Utf8).alias("sn")).drop_nulls("sn")
    P = P.select("ADEP_mvt", "sn", "lat", "lon", "month")
    P.write_parquet(RUNS)
    return P


def map_rule(P: pl.DataFrame) -> pl.DataFrame:
    """stand_map.build_map's rule on a set of stand runs."""
    M = P.group_by("ADEP_mvt", "sn").agg(pl.len().alias("n"), pl.col("lat").median().alias("mlat"),
                                          pl.col("lon").median().alias("mlon"))
    Q = P.join(M, on=["ADEP_mvt", "sn"])
    Q = Q.with_columns(pl.Series("dm", A._dist(*Q.select("lat", "lon", "mlat", "mlon").to_numpy().T)))
    M = M.join(Q.group_by("ADEP_mvt", "sn").agg(pl.col("dm").median().alias("spread")), on=["ADEP_mvt", "sn"])
    return (M.filter((pl.col("n") >= SM.MIN_N) & (pl.col("spread") <= SM.MAX_SPREAD))
            .select("ADEP_mvt", "sn", pl.col("mlat").alias("slat"), pl.col("mlon").alias("slon")).sort("ADEP_mvt", "sn"))


def tables() -> dict[str, dict[str, pl.DataFrame]]:
    """{month: {"fill12": table, "rand12": table}} for the 12 months of 2025 and "2026"."""
    P = runs()
    base = SM.stand_table("fill")
    out = {}
    for m in [f"2025-{i:02d}" for i in range(1, 13)] + ["2026"]:
        new = map_rule(P if m == "2026" else P.filter(pl.col("month") != m)).join(base, on=["ADEP_mvt", "sn"], how="anti")
        rng = np.random.default_rng(A.SEED)
        parts = []
        for a in sorted(new["ADEP_mvt"].unique().to_list()):
            x = new.filter(pl.col("ADEP_mvt") == a)
            perm = rng.permutation(x.height)
            parts.append(x.with_columns(pl.Series("slat", x["slat"].to_numpy()[perm]),
                                        pl.Series("slon", x["slon"].to_numpy()[perm])))
        rand = pl.concat(parts) if parts else new
        out[m] = {"fill12": pl.concat([base, new]), "rand12": pl.concat([base, rand])}
        print(f"{m}: {new.height} stands added to fill ({dict(new.group_by('ADEP_mvt').len().sort('ADEP_mvt').iter_rows())})",
              flush=True)
    return out


def build() -> None:
    T = tables()
    for key in KEYS:
        have = [d for d in DAYS[key] if os.path.exists(os.path.join(s1.SCRATCH, f"{L.tag_of(d)}_joined.parquet"))]
        for v in ("fill12", "rand12") if key != "2026" else ("fill12",):
            fr = []
            for d in have:
                p = os.path.join(L.OUT, f"appear_{v}_{d}.parquet")
                if not os.path.exists(p):
                    A.day_appear(d, T[month_of(d)][v]).write_parquet(p + ".tmp")
                    os.replace(p + ".tmp", p)
                fr.append(pl.read_parquet(p))
            D = pl.concat(fr)
            assert D["MVT_ID_mvt"].n_unique() == D.height
            D.write_parquet(os.path.join(L.OUT, f"appear_{v}_{key}.parquet"))
            print(f"{v} {key}: {D.height} rows, own stand {float((D['own_ok'] == 1).mean()):.3f}, at own stand "
                  f"{float((D['d_dep1_own'] <= A.R_IN).fill_null(False).mean()):.3f}", flush=True)


if __name__ == "__main__":
    build()
