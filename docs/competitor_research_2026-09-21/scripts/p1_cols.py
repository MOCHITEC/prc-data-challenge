"""P1 §1.3 [S-new]: what model B actually sees on an unmatched row.

Read-only null-rate scan of the cached feature frames. Writes nothing.
    .venv/bin/python docs/competitor_research_2026-09-21/scripts/p1_cols.py
"""

import polars as pl

CATS = [
    "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_mvt",
    "ADES_REGION", "FLIGHT_mvt", "FLIGHT_RULE_mvt",
]

for path, name in (("data/work/feat_train.parquet", "train"), ("data/work/feat_rank.parquet", "rank")):
    df = pl.read_parquet(path, columns=CATS + ["mvt_minus_aobt3"])
    u = df.filter(pl.col("mvt_minus_aobt3").is_null())
    print(name, "unmatched n =", u.height)
    print("  ", {c: round(u[c].null_count() / u.height, 4) for c in CATS})
    if name == "train":
        li = u.filter(pl.col("ADEP_mvt") == "LIRF")
        print("  LIRF unmatched n =", li.height,
              {c: round(li[c].null_count() / li.height, 4) for c in CATS})
        print("  LIRF unmatched distinct FLIGHT_mvt prefix-3:",
              li.select(pl.col("FLIGHT_mvt").str.slice(0, 3)).n_unique())
