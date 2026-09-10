"""Submission gate: a file must pass every check here before it is uploaded (docs/01 §8, U10).

Checks: exact schema match with the official template, identical MVT_ID set and order, expected row
count, no null/NaN/inf, non-negative values, plausible range."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import EXPECTED_SUBMISSION_ROWS, SUBMISSION_ID_COL, SUBMISSION_TARGET_COL

MAX_PLAUSIBLE_SECONDS = 3 * 3600  # sanity bound, not a clipping rule


@dataclass
class GateResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Submission gate failed:\n- " + "\n- ".join(self.errors))


def validate(
    submission: pa.Table, template: pa.Table, expected_rows: int | None = EXPECTED_SUBMISSION_ROWS
) -> GateResult:
    res = GateResult(ok=True, n_rows=submission.num_rows)

    # 1) schema: same column names, order and types as the template
    if submission.schema.names != template.schema.names:
        res.errors.append(f"column names/order differ: {submission.schema.names} vs {template.schema.names}")
    for name in template.schema.names:
        if name in submission.schema.names and submission.schema.field(name).type != template.schema.field(name).type:
            res.errors.append(
                f"dtype of {name} differs: {submission.schema.field(name).type} vs template {template.schema.field(name).type}"
            )

    # 2) row count
    if submission.num_rows != template.num_rows:
        res.errors.append(f"row count {submission.num_rows} != template {template.num_rows}")
    if expected_rows is not None and template.num_rows != expected_rows:
        res.warnings.append(f"template has {template.num_rows} rows, expected {expected_rows}")

    if SUBMISSION_ID_COL in submission.schema.names and SUBMISSION_ID_COL in template.schema.names:
        ids_s, ids_t = submission[SUBMISSION_ID_COL], template[SUBMISSION_ID_COL]
        # 3) identical id multiset and identical order (order is not required by the rules, but keep it)
        if pc.count_distinct(ids_s).as_py() != submission.num_rows:
            res.errors.append("duplicate MVT_ID_mvt in submission")
        if submission.num_rows == template.num_rows and not ids_s.equals(ids_t):
            same_set = set(ids_s.to_pylist()) == set(ids_t.to_pylist())
            (res.warnings if same_set else res.errors).append(
                "MVT_ID_mvt order differs from template" if same_set else "MVT_ID_mvt set differs from template"
            )

    # 4) target values
    if SUBMISSION_TARGET_COL in submission.schema.names:
        y = submission[SUBMISSION_TARGET_COL]
        n_null = pc.sum(pc.is_null(y, nan_is_null=True)).as_py() or 0
        if n_null:
            res.errors.append(f"{n_null} null/NaN values in {SUBMISSION_TARGET_COL}")
        if pa.types.is_floating(y.type):
            finite = pc.is_finite(y)
            n_inf = pc.sum(pc.invert(pc.fill_null(finite, True))).as_py() or 0
            if n_inf:
                res.errors.append(f"{n_inf} non-finite values in {SUBMISSION_TARGET_COL}")
        vals = pc.drop_null(y)
        if len(vals):
            mn, mx = pc.min(vals).as_py(), pc.max(vals).as_py()
            if mn is not None and mn < 0:
                res.errors.append(f"negative prediction found (min={mn})")
            if mx is not None and (isinstance(mx, float) and math.isnan(mx) or mx > MAX_PLAUSIBLE_SECONDS):
                res.warnings.append(f"max prediction {mx} exceeds {MAX_PLAUSIBLE_SECONDS}s; check outliers")

    res.ok = not res.errors
    return res


def validate_files(submission_path: Path, template_path: Path) -> GateResult:
    return validate(pq.read_table(submission_path), pq.read_table(template_path))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("usage: python -m prc.validate_submission <submission.parquet> <submitting.parquet>")
        return 2
    res = validate_files(Path(argv[0]), Path(argv[1]))
    for w in res.warnings:
        print(f"WARNING: {w}")
    for e in res.errors:
        print(f"ERROR: {e}")
    print("OK" if res.ok else "FAILED", f"({res.n_rows} rows)")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
