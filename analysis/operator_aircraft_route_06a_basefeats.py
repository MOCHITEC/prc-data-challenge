"""operator_aircraft_route 06a: repo baseline features (prc.features.build_departure_features), per month file.

Month-by-month build (context windows lose < 1h at each month edge; acceptable for an A/B of new features since
both arms share it). Output: data/interim/oar_basefeats_train.parquet (float32, a curated subset of columns).
"""

from __future__ import annotations

import glob
import os
import sys

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from prc.features import build_departure_features  # noqa: E402

KEEP_CAT = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
            "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_mvt", "ADES_REGION"]
DROP_PREFIX = ("dep_tko_next_", "arr_land_next_")  # keep the file compact; look-ahead counts stay in via 60m ones


def main() -> None:
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "training_*.parquet"))):
        mv = pl.read_parquet(f).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("FLIGHT_ID_mvt").cast(pl.Int64))
        ft = build_departure_features(mv)
        num = [c for c, t in ft.schema.items() if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32)
               and c not in ("MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt") and not c.endswith("_slog")
               and not (c.startswith(DROP_PREFIX) and not c.endswith("60m"))]
        ft = ft.select(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"] + KEEP_CAT +
                       [pl.col(c).cast(pl.Float32) for c in num])
        out.append(ft)
        print(os.path.basename(f), ft.shape, flush=True)
    df = pl.concat(out, how="diagonal_relaxed")
    df.write_parquet(os.path.join(ROOT, "data", "interim", "oar_basefeats_train.parquet"))
    print(df.shape)


if __name__ == "__main__":
    main()
