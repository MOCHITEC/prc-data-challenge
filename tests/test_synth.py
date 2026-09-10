import polars as pl

from prc.schema import ALL_COLS
from prc.synth import generate_month, write_synthetic_dataset


def test_month_schema_and_identity():
    df = generate_month(2025, 1, seed=1, deps_per_airport_day=5)
    assert df.columns == ALL_COLS
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    diff = (dep["MVT_TIME_UTC_mvt"] - dep["BLOCK_TIME_UTC_mvt"]).dt.total_seconds() - dep[
        "TAXITIME_SEC_mvt"
    ]
    assert diff.abs().max() <= 1
    assert dep["FLIGHT_ID_mvt"].null_count() > 0  # unmatched rows exist


def test_dataset_files(tmp_path):
    paths = write_synthetic_dataset(tmp_path, months=[(2025, 1), (2025, 7)], deps_per_airport_day=3)
    rank = pl.read_parquet(paths["ranking.parquet"])
    dep = rank.filter(pl.col("PHASE_mvt") == "DEP")
    assert dep["BLOCK_TIME_UTC_mvt"].null_count() == dep.height
    assert dep["TAXITIME_SEC_mvt"].null_count() == dep.height
    assert rank.filter(pl.col("PHASE_mvt") == "ARR")["TAXITIME_SEC_mvt"].null_count() == 0
    sub = pl.read_parquet(paths["submitting.parquet"])
    assert sub.columns == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] and sub.height == dep.height
