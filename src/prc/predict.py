"""Fit on all labelled 2025 departures and produce a ranking prediction + submission parquet (docs/README)."""

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
    """Predict the ranking frame after checking feature availability parity (docs/README)."""
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


def _align_id_dtype(tmpl: pl.DataFrame, pred: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Join the template and the predictions on a common id dtype, refusing a lossy cast.

    The official files type MVT_ID_mvt as float64 holding integral values; a frame that has been through an
    integer round trip comes back as int64. polars will not join across the two, and silently casting either
    way is only safe while the values stay integral and inside float64's exact range."""
    t_type, p_type = tmpl[SUBMISSION_ID_COL].dtype, pred[SUBMISSION_ID_COL].dtype
    if t_type == p_type:
        return tmpl, pred
    for name, frame in (("template", tmpl), ("prediction", pred)):
        col = frame[SUBMISSION_ID_COL]
        if col.dtype.is_float():
            if (col - col.round()).abs().max() != 0.0:
                raise ValueError(f"{name} {SUBMISSION_ID_COL} holds non-integral values; refusing to cast")
            if col.abs().max() > 2**53:
                raise ValueError(f"{name} {SUBMISSION_ID_COL} exceeds float64's exact integer range")
    cast = pl.col(SUBMISSION_ID_COL).cast(pl.Int64)
    return tmpl.with_columns(cast), pred.with_columns(cast)


def write_submission(
    pred: pl.DataFrame, template_path: Path, out_path: Path, reference_median: float | None = None
) -> pa.Table:
    """Fill the official template in its own row order and dtype; run the gate; write parquet.

    reference_median is the median of the labels the model was trained on; the gate refuses a submission whose
    own median has drifted far from it, which is the only check that notices a model trained on the wrong data.

    Ids and values are taken from the SAME frame. Reading ids from the template and values from the join result
    is position-dependent, and polars does not guarantee join order, so that pairing could silently ship
    shuffled predictions past a gate that only checks the id set."""
    template = pq.read_table(template_path)
    tmpl = pl.from_arrow(template.select([SUBMISSION_ID_COL]))
    tmpl, pred = _align_id_dtype(tmpl, pred)
    joined = tmpl.join(pred, on=SUBMISSION_ID_COL, how="left", maintain_order="left")
    if joined.height != template.num_rows:
        raise ValueError(f"join produced {joined.height} rows for a {template.num_rows}-row template")
    if joined["pred"].null_count():
        raise ValueError(f"{joined['pred'].null_count()} template ids without prediction")

    id_type = template.schema.field(SUBMISSION_ID_COL).type
    ids = pa.array(joined[SUBMISSION_ID_COL].to_numpy()).cast(id_type)
    if not ids.equals(template[SUBMISSION_ID_COL].combine_chunks()):
        raise ValueError("submission ids are not in template order")

    target_type = template.schema.field(SUBMISSION_TARGET_COL).type
    vals = joined["pred"].to_numpy().astype(np.float64)
    note: str | None = None
    schema = template.schema
    if pa.types.is_integer(target_type):
        # docs/README: the official dtype is unconfirmed. A safe cast of 600.4 to int64 raises, so round first.
        rounded = np.rint(vals)
        note = f"rounded to {target_type} for the template dtype (max shift {np.abs(rounded - vals).max():.3f}s)"
        values = pa.array(rounded, type=pa.float64()).cast(target_type)
    elif pa.types.is_null(target_type):
        # an all-null template column is typed null in parquet and cannot hold numbers; promote to float64
        note = "template target column is null-typed; written as float64"
        values = pa.array(vals, type=pa.float64())
        schema = pa.schema([template.schema.field(SUBMISSION_ID_COL), pa.field(SUBMISSION_TARGET_COL, pa.float64())])
    else:
        values = pa.array(vals, type=pa.float64()).cast(target_type)

    sub = pa.table({SUBMISSION_ID_COL: ids, SUBMISSION_TARGET_COL: values}, schema=schema)
    if schema.equals(template.schema):
        gate_template = template
    else:  # promoted target dtype: keep the template's ids and row count so the gate still checks them
        gate_template = pa.table(
            {SUBMISSION_ID_COL: ids, SUBMISSION_TARGET_COL: pa.nulls(template.num_rows, type=pa.float64())},
            schema=schema,
        )
    res = validate(sub, gate_template, expected_rows=None, reference_median=reference_median)
    if note:
        res.warnings.append(note)
    res.raise_if_failed()
    for w in res.warnings:
        print(f"submission gate warning: {w}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out_path)
    return sub


def log_submission(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
