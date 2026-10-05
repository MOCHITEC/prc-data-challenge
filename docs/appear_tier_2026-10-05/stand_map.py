"""Stand positions from 2026 ADS-B and the appear columns rebuilt on them (plan_standmap.md).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/stand_map.py map      # -> stand_map_2026
    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/stand_map.py build    # appear_<v>_<key>

The map: for every 2026 ranking departure with a callsign ADS-B join and a stage-1 stand run of >= 10 min
(`stand_dur` >= 600 s), the position of the run's last fix (t_move_lo); per (airport, normalised STAND_mvt) the median
latitude / longitude, kept when >= 5 departures and the median distance to that point is <= 40 m. Label-free and
2026 only, so no 2025 row helps place its own stand.

Variants of the own-stand table (adsb_appear.day_appear takes it as ST; everything else in the appear columns is
unchanged): osm (as before), fill (OSM, else map), mapfirst (map, else OSM), rand (mapfirst with the map stands'
coordinates permuted among the map stands of the same airport: same stands filled, wrong positions)."""

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_appear as A  # noqa: E402
import adsb_learn_features as L  # noqa: E402
import adsb_stage1_feasibility as s1  # noqa: E402
from build_features import DAYS  # noqa: E402

from prc.osm import normalise_stand  # noqa: E402

MIN_DUR, MIN_N, MAX_SPREAD = 600.0, 5, 40.0
VARIANTS = ("osm", "fill", "mapfirst", "rand")
MAP = os.path.join(L.OUT, "stand_map_2026.parquet")


def build_map() -> pl.DataFrame:
    rows = []
    for d in L.DAYS["2026"]:
        jp = os.path.join(s1.SCRATCH, f"{L.tag_of(d)}_joined.parquet")
        if not os.path.exists(jp):
            continue
        J = pl.read_parquet(jp, columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "icao", "join", "t_move_lo", "stand_dur"])
        J = J.filter(pl.col("icao").is_not_null() & pl.col("join").is_in(["cs", "fn"]) & (pl.col("stand_dur") >= MIN_DUR)
                     & pl.col("t_move_lo").is_not_null())
        fx = pl.read_parquet(os.path.join(s1.SCRATCH, f"{L.tag_of(d)}.parquet"), columns=["icao", "apt", "t", "lat", "lon"])
        P = J.join(fx.rename({"apt": "ADEP_mvt", "t": "t_move_lo"}), on=["icao", "ADEP_mvt", "t_move_lo"], how="inner")
        rows.append(P.unique("MVT_ID_mvt").select("ADEP_mvt", "STAND_mvt", "lat", "lon"))
    P = pl.concat(rows).with_columns(
        pl.col("STAND_mvt").cast(pl.Utf8).map_elements(normalise_stand, return_dtype=pl.Utf8).alias("sn")).drop_nulls("sn")
    M = P.group_by("ADEP_mvt", "sn").agg(pl.len().alias("n"), pl.col("lat").median().alias("mlat"),
                                          pl.col("lon").median().alias("mlon"))
    Q = P.join(M, on=["ADEP_mvt", "sn"])
    Q = Q.with_columns(pl.Series("dm", A._dist(*Q.select("lat", "lon", "mlat", "mlon").to_numpy().T)))
    M = M.join(Q.group_by("ADEP_mvt", "sn").agg(pl.col("dm").median().alias("spread")), on=["ADEP_mvt", "sn"])
    M = M.filter((pl.col("n") >= MIN_N) & (pl.col("spread") <= MAX_SPREAD)).sort("ADEP_mvt", "sn")
    M.write_parquet(MAP)
    print(f"stand map: {M.height} stands ->", MAP, dict(M.group_by("ADEP_mvt").len().sort("ADEP_mvt").iter_rows()))
    return M


def stand_table(variant: str) -> pl.DataFrame:
    osm = A.stands()
    if variant == "osm":
        return osm
    m = pl.read_parquet(MAP).select("ADEP_mvt", "sn", pl.col("mlat").alias("slat"), pl.col("mlon").alias("slon"))
    if variant == "fill":
        return pl.concat([osm, m.join(osm, on=["ADEP_mvt", "sn"], how="anti")])
    if variant == "rand":
        rng = np.random.default_rng(A.SEED)
        parts = []
        for a in sorted(m["ADEP_mvt"].unique().to_list()):
            x = m.filter(pl.col("ADEP_mvt") == a)
            perm = rng.permutation(x.height)
            parts.append(x.with_columns(pl.Series("slat", x["slat"].to_numpy()[perm]),
                                        pl.Series("slon", x["slon"].to_numpy()[perm])))
        m = pl.concat(parts)
    assert variant in ("mapfirst", "rand"), variant
    return pl.concat([m, osm.join(m, on=["ADEP_mvt", "sn"], how="anti")])


def build() -> None:
    for v in VARIANTS:
        ST = stand_table(v)
        for key in ("2025", "2025nd", "2026"):
            have = [d for d in DAYS[key] if os.path.exists(os.path.join(s1.SCRATCH, f"{L.tag_of(d)}_joined.parquet"))]
            fr = []
            for d in have:
                p = os.path.join(L.OUT, f"appear_{v}_{d}.parquet")
                if not os.path.exists(p):
                    A.day_appear(d, ST).write_parquet(p)
                fr.append(pl.read_parquet(p))
            D = pl.concat(fr)
            assert D["MVT_ID_mvt"].n_unique() == D.height
            D.write_parquet(os.path.join(L.OUT, f"appear_{v}_{key}.parquet"))
            print(f"{v} {key}: {D.height} rows, own stand {float((D['own_ok'] == 1).mean()):.3f}, at own stand "
                  f"{float((D['d_dep1_own'] <= A.R_IN).fill_null(False).mean()):.3f}", flush=True)


if __name__ == "__main__":
    {"map": build_map, "build": build}[sys.argv[1]]()
