"""Fit on all labelled 2025 departures and produce a ranking prediction + submission parquet (docs/04 §8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL
from .train import TrainConfig, feature_parity_report, predict_fold
from .validate_submission import validate

PARITY_GAP_LIMIT = 0.20


def predict_ranking(
    train_feats: pl.DataFrame, rank_feats: pl.DataFrame, cfg: TrainConfig, parity_gap_limit: float = PARITY_GAP_LIMIT
) -> tuple[pl.DataFrame, dict]:
    """Predict the ranking frame after checking feature availability parity (docs/04 section 3)."""
    parity = feature_parity_report(train_feats, rank_feats, cfg)
    bad = parity.filter(pl.col("gap") > parity_gap_limit)
    if bad.height:
        raise ValueError(
            "feature availability differs between training and ranking; fix or drop these features:\n"
            + str(bad.select(["feature", "null_rate_train", "null_rate_target", "gap"]))
        )
    pred, info = predict_fold(train_feats, rank_feats, cfg)
    info["parity_max_gap"] = float(parity["gap"].max()) if parity.height else 0.0
    return pred, info


def write_submission(pred: pl.DataFrame, template_path: Path, out_path: Path) -> pa.Table:
    """Fill the official template in its own row order and dtype; run the gate; write parquet."""
    template = pq.read_table(template_path)
    tmpl = pl.from_arrow(template.select([SUBMISSION_ID_COL]))
    joined = tmpl.join(pred, on="MVT_ID_mvt", how="left")
    if joined["pred"].null_count():
        raise ValueError(f"{joined['pred'].null_count()} template ids without prediction")
    target_type = template.schema.field(SUBMISSION_TARGET_COL).type
    values = pa.array(joined["pred"].to_numpy().astype(np.float64), type=pa.float64()).cast(target_type)
    sub = pa.table(
        {SUBMISSION_ID_COL: template[SUBMISSION_ID_COL], SUBMISSION_TARGET_COL: values}, schema=template.schema
    )
    res = validate(sub, template, expected_rows=None)
    res.raise_if_failed()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out_path)
    return sub


def log_submission(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
