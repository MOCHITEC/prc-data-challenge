"""Month-scoped versions of two frame-scoped columns (docs/research_2026-10-05_drift/plan.md).

`fl_nm_rate_other` and `stand_rwy_share` are shares counted over the whole frame they are built on. The training
frame is twelve months and the ranking frame two, so the same quantity has a different resolution in the two
frames: a flight number seen 300 times in 2025 gets a near-continuous rate, the same flight seen 50 times in
2026 a coarse one. A single-column classifier tells the frames apart from `fl_nm_rate_other` alone with AUC 0.747
(stand_rwy_share 0.686). Counting both within the calendar month gives the two frames the same scope
(AUC 0.521 and 0.620; the rest of stand_rwy_share's drift is the 2026 runway mix, which is real). Label-free."""

from __future__ import annotations

import polars as pl

MONTH_SCOPE_COLS = ("fl_nm_rate_other", "stand_rwy_share")


def month_scope_columns(df: pl.DataFrame) -> pl.DataFrame:
    """`df` with MONTH_SCOPE_COLS recomputed within (calendar month of MVT_TIME_UTC_mvt). Row order unchanged."""
    ym = pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y%m")
    m = pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Float64)
    g = [ym, pl.col("ADEP_mvt"), pl.col("FLIGHT_mvt")]
    n_other = pl.len().over(g) - 1
    rate = (
        pl.when(pl.col("FLIGHT_mvt").is_not_null() & (n_other > 0))
        .then((m.sum().over(g) - m) / n_other)
        .otherwise(None)
    )
    share = pl.len().over([ym, pl.col("ADEP_mvt"), pl.col("STAND_mvt"), pl.col("RUNWAY_mvt")]) / pl.len().over(
        [ym, pl.col("ADEP_mvt"), pl.col("STAND_mvt")]
    )
    out = df.with_columns(
        rate.cast(df.schema.get("fl_nm_rate_other", pl.Float32)).alias("fl_nm_rate_other"),
        share.cast(df.schema.get("stand_rwy_share", pl.Float64)).alias("stand_rwy_share"),
    )
    return out
