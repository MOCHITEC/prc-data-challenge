"""operator_aircraft_route 06b: label-free new features for training and ranking departures.

Needs 00, 02 and 04 to have run (oar_dep / oar_route / oar_turn). Output data/interim/oar_newfeats.parquet.
All inputs exist in ranking.parquet (DEP identity/time columns, ARR in-block times, external airport coordinates).
"""

from __future__ import annotations

import os

import polars as pl
from operator_aircraft_route_common import INTERIM, flight_format, load_dep


def main() -> None:
    d = load_dep(["MVT_ID_mvt", "split", "FLIGHT_mvt", "CALLSIGN_flt", "airline", "ADEP_mvt"])
    d = d.with_columns(flight_format().alias("flight_fmt"))
    turn = pl.read_parquet(os.path.join(INTERIM, "oar_turn.parquet"))
    route = pl.read_parquet(os.path.join(INTERIM, "oar_route.parquet"))
    d = d.join(turn, on="MVT_ID_mvt", how="left").join(route, on="MVT_ID_mvt", how="left")
    d = d.with_columns(
        # another aircraft blocked in at this stand after our NM off-block -> we had left by then: y >= ta_gap
        pl.when(pl.col("ta_aobt3_minus_inblock") < 0).then(pl.col("ta_gap")).otherwise(None).alias("stand_lb"),
        pl.col("wide").cast(pl.Int8), pl.col("heli").cast(pl.Int8),
    )
    d.drop("CALLSIGN_flt", "FLIGHT_mvt", "ADEP_mvt").write_parquet(os.path.join(INTERIM, "oar_newfeats.parquet"))
    print(d.group_by("split").agg([pl.col(c).is_null().mean().alias(c) for c in
                                  ["airline", "ta_gap", "stand_lb", "dist_km", "ta_linked"]]))
    print(d.group_by("split", "flight_fmt").len().sort("split", "len"))


if __name__ == "__main__":
    main()
