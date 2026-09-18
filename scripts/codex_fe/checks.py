"""Executable checks of screening arithmetic, temporal fitting and feature availability."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
import json
from pathlib import Path

import numpy as np
import polars as pl
from candidates import build
from screen import category, correction, evaluate


def main():
    df = pl.DataFrame(
        {
            "MVT_ID_mvt": [1, 2, 3, 4],
            "ADEP_mvt": ["A", "A", "B", "B"],
            "unmatched": [0, 0, 1, 1],
            "x": [0.0, 1.0, 2.0, 3.0],
            "residual": [10.0, 10.0, 20.0, 20.0],
            "TAXITIME_SEC_mvt": [10.0, 20.0, 30.0, 40.0],
            "pred": [12.0, 22.0, 32.0, 42.0],
            "MVT_TIME_UTC_mvt": pl.datetime_range(
                pl.datetime(2025, 10, 1), pl.datetime(2025, 10, 4), interval="1d", eager=True
            ),
        }
    )
    same = evaluate(df, df["pred"].to_numpy())
    assert same["delta"] == 0 and same["ci"] == [0, 0]
    perfect = evaluate(df, df["TAXITIME_SEC_mvt"].to_numpy())
    assert perfect["delta"] == -2 and perfect["ci"] == [-2, -2]
    changed = df.with_columns(pl.lit(1e9).alias("residual"))
    np.testing.assert_array_equal(correction(df, df, "x"), correction(df, changed, "x"))
    unseen = df.with_columns(pl.lit("C").alias("ADEP_mvt"))
    np.testing.assert_array_equal(correction(df, unseen, "x"), np.zeros(4))
    a, _ = category(df, df, "x")
    b, _ = category(df, changed.with_columns(pl.lit(1e20).alias("x")), "x")
    np.testing.assert_array_equal(a, b)
    f = pl.read_parquet("data/work/feat_train.parquet").head(1000)
    a = (
        pl.scan_parquet(sorted(Path("data/raw").glob("training_*.parquet")))
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            "ADES_mvt",
            "STAND_mvt",
            "BLOCK_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt",
            "AIRCRAFT_TYPE_mvt",
            "SCHED_TIME_UTC_mvt",
            "MVT_TIME_UTC_mvt",
        )
        .limit(2000)
        .collect()
    )
    first = build(f, a)
    masked = build(
        f.with_columns(
            pl.lit(None).cast(f.schema["BLOCK_TIME_UTC_mvt"]).alias("BLOCK_TIME_UTC_mvt"),
            pl.lit(None).cast(pl.Int32).alias("TAXITIME_SEC_mvt"),
        ),
        a,
    )
    assert first.equals(masked), "departure labels influenced candidate features"
    train = pl.read_parquet("outputs/codex_fe/candidates_train.parquet")
    rank = pl.read_parquet("outputs/codex_fe/candidates_rank.parquet")
    assert train.columns == rank.columns and train.schema == rank.schema
    assert train["MVT_ID_mvt"].n_unique() == train.height
    assert rank["MVT_ID_mvt"].n_unique() == rank.height
    parity = [
        {
            "feature": c,
            "train_null": float(train[c].null_count() / train.height),
            "rank_null": float(rank[c].null_count() / rank.height),
        }
        for c in train.columns
        if c != "MVT_ID_mvt"
    ]
    Path("docs/codex_fe_results/parity.json").write_text(json.dumps(parity, indent=2))
    print(
        "PASS: exact paired-bootstrap arithmetic; evaluation residual isolation; unseen cells; train-only cuts; departure-label masking; ID uniqueness; train/rank schema parity."
    )


if __name__ == "__main__":
    main()
