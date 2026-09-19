"""Verify production gap equals the frozen experiment on every train/ranking row."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
import json
from pathlib import Path

import polars as pl

from prc.features import revision_context_features, stand_arrival_features
from prc.train import TrainConfig, visible_numeric_columns

cols = [
    "stand_arrival_gap",
    "stand_aobt_turn",
    "stand_arrival_type_match",
    "eobt_schedule_revision",
    "delay_neighbor_deviation",
]
results = []
for split in ["train", "rank"]:
    f = pl.read_parquet(
        f"data/work/feat_{split}.parquet",
        columns=[
            "MVT_ID_mvt",
            "ADEP_mvt",
            "STAND_mvt",
            "MVT_TIME_UTC_mvt",
            "AIRCRAFT_TYPE_mvt",
            "AOBT_3_flt",
            "EOBT_1_flt",
            "SCHED_TIME_UTC_mvt",
            "aobt3_minus_eobt1",
            "nbr_aobt_delay_mean_30m",
        ],
    )
    paths = (
        sorted(Path("data/raw").glob("training_*.parquet")) if split == "train" else [Path("data/raw/ranking.parquet")]
    )
    a = (
        pl.scan_parquet(paths)
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select("ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt")
        .collect()
    )
    got = (
        stand_arrival_features(f, a)
        .join(revision_context_features(f, f), on="MVT_ID_mvt", maintain_order="left")
        .select("MVT_ID_mvt", *cols)
    )
    expected = pl.read_parquet(f"outputs/codex_fe/candidates_{split}.parquet", columns=["MVT_ID_mvt", *cols])
    assert got.equals(expected)
    assert not set(cols) & set(visible_numeric_columns(got, TrainConfig()))
    assert set(cols) <= set(visible_numeric_columns(got, TrainConfig(include_candidates=cols)))
    results.append(
        {
            "split": split,
            "rows": f.height,
            "exact_equality": True,
            "columns": cols,
            "default_hidden": True,
            "opt_in_visible": True,
        }
    )
Path("docs/codex_fe_results/pipeline_parity.json").write_text(json.dumps(results, indent=2))
print(results)
