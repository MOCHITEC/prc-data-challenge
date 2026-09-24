"""Build the baseline feature table (existing src/prc/features.py, unchanged) for training departures, one airport
at a time to stay within the memory budget. Output: SCRATCH/base_<APT>.parquet.

This mirrors the production feature path so the stand features in stand_03_model.py are measured *on top of*
what the current model already has (clock deltas, runway headways, window counts, in-system queue ...).
"""

from __future__ import annotations

import glob
import os
import sys

from stand_common import RAW, SCRATCH

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import polars as pl  # noqa: E402

from prc.features import build_departure_features, numeric_feature_columns  # noqa: E402
from prc.schema import ALL_COLS  # noqa: E402

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
CAT_KEEP = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt"]


def main() -> None:
    files = sorted(glob.glob(os.path.join(RAW, "training_*.parquet")))
    lf = pl.concat([pl.scan_parquet(f).select(ALL_COLS) for f in files])
    for apt in APTS:
        out = f"{SCRATCH}/base_{apt}.parquet"
        if os.path.exists(out):
            continue
        mv = lf.filter((pl.col("ADEP_mvt") == apt) | (pl.col("ADES_mvt") == apt)).collect()
        # departures of this airport + arrivals into it (context features need both)
        mv = mv.filter(
            ((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt))
            | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt))
        )
        f = build_departure_features(mv)
        num = [c for c in numeric_feature_columns(f) if not c.endswith("_slog")]
        keep = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
        keep += [c for c in CAT_KEEP if c in f.columns]
        extra = mv.filter(pl.col("PHASE_mvt") == "DEP").select(
            ["MVT_ID_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt"]
        )
        f = f.select([c for c in keep if c in f.columns] + [c for c in num if c not in keep])
        f = f.join(extra.drop([c for c in extra.columns if c in f.columns and c != "MVT_ID_mvt"]), on="MVT_ID_mvt", how="left")
        f = f.with_columns([pl.col(c).cast(pl.Float32) for c in num if f.schema[c] in (pl.Float64,)])
        f.write_parquet(out)
        print(apt, f.shape, flush=True)
        del mv, f


if __name__ == "__main__":
    main()
