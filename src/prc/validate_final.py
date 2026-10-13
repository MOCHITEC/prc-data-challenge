"""Final-phase gate for `<team>_final.parquet`: run before the one final upload (docs/final_phase_2026-10-11).

    python -m prc.validate_final <file> --template final_submitting.parquet --ranking final_ranking.parquet \
        [--expect-md5 MD5] [--json report.json]

Hard failures (exit 1): schema (names, order, dtypes) differs from the template; row count differs; ids differ from
the template in set or order; duplicate ids; null/NaN/inf; negative values; a constant vector; an id missing from
final_ranking; a month other than Jan/Feb/Jun/Jul 2026, or one of them empty; the md5 differs from --expect-md5.

Reported, not failed: per-month row counts and prediction quantiles, per airport per month row counts, the max and
the number of rows above 10,800 s, the md5. Without --ranking the month and airport checks cannot run, and that is a
hard failure too, because the final is scored on four months and a month mix-up is the failure this gate exists for.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL, TARGET_AIRPORTS

FINAL_MONTHS = {(2026, 1): "Jan", (2026, 2): "Feb", (2026, 6): "Jun", (2026, 7): "Jul"}
EXPECTED_FINAL_ROWS = 670_790  # final_submitting.parquet, 2026-10-08 (beautiful-davinci b53bdaa; re-counted 10-13)
LONG_SECONDS = 10_800  # 3 h: reported, not a clipping rule
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # an integrity fingerprint to compare with the hand-off, not a security hash
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class FinalReport:
    ok: bool = True
    file: str = ""
    md5: str = ""
    n_rows: int = 0
    template_rows: int = 0
    schema: list[str] = field(default_factory=list)
    template_schema: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_null: int = 0
    n_negative: int = 0
    max_value: float | None = None
    min_value: float | None = None
    n_above_long: int = 0
    # "Jan" -> {"rows", "template_rows", "quantiles", "mean", "n_above_10800"}
    months: dict = field(default_factory=dict)
    airport_month: dict = field(default_factory=dict)  # airport -> {"Jan": n, ...}

    def fail(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("final gate failed:\n- " + "\n- ".join(self.errors))

    def summary(self) -> dict:
        """The part that goes into the submissions ledger."""
        return {
            "ok": self.ok,
            "md5": self.md5,
            "n_rows": self.n_rows,
            "errors": self.errors,
            "warnings": self.warnings,
            "max_value": self.max_value,
            "n_above_10800": self.n_above_long,
            "months": {m: v["rows"] for m, v in self.months.items()},
        }


def _schema_strs(schema: pa.Schema) -> list[str]:
    return [f"{f.name}: {f.type}" for f in schema]


def validate_final(
    submission: pa.Table,
    template: pa.Table,
    ranking: pa.Table | None,
    md5: str = "",
    expect_md5: str | None = None,
    expected_rows: int | None = EXPECTED_FINAL_ROWS,
) -> FinalReport:
    rep = FinalReport(md5=md5, n_rows=submission.num_rows, template_rows=template.num_rows)
    rep.schema, rep.template_schema = _schema_strs(submission.schema), _schema_strs(template.schema)

    # md5: the file must be the one the hand-off names
    if expect_md5 is not None and md5 != expect_md5.strip().lower():
        rep.fail(f"md5 {md5} != expected {expect_md5}")

    # schema: names, order and types exactly as the template's
    if submission.schema.names != template.schema.names:
        rep.fail(f"column names/order differ: {submission.schema.names} vs template {template.schema.names}")
    for name in template.schema.names:
        if name not in submission.schema.names:
            continue
        st, tt = submission.schema.field(name).type, template.schema.field(name).type
        if st != tt:
            rep.fail(f"dtype of {name} is {st}, template has {tt}")

    # rows
    if submission.num_rows != template.num_rows:
        rep.fail(f"row count {submission.num_rows:,} != template {template.num_rows:,}")
    if expected_rows is not None and template.num_rows != expected_rows:
        rep.warnings.append(f"template has {template.num_rows:,} rows; {expected_rows:,} expected")

    have_ids = SUBMISSION_ID_COL in submission.schema.names and SUBMISSION_ID_COL in template.schema.names
    if have_ids:
        ids_s = submission[SUBMISSION_ID_COL].combine_chunks()
        ids_t = template[SUBMISSION_ID_COL].combine_chunks()
        if ids_s.null_count:
            rep.fail(f"{ids_s.null_count} null {SUBMISSION_ID_COL}")
        if pc.count_distinct(ids_s).as_py() != len(ids_s):
            rep.fail(f"duplicate {SUBMISSION_ID_COL} in the file")
        if not ids_s.equals(ids_t):
            same_set = len(ids_s) == len(ids_t) and set(ids_s.to_pylist()) == set(ids_t.to_pylist())
            rep.fail(
                f"{SUBMISSION_ID_COL} order differs from the template (same set)"
                if same_set
                else f"{SUBMISSION_ID_COL} set differs from the template"
            )
    else:
        rep.fail(f"{SUBMISSION_ID_COL} missing")

    if SUBMISSION_TARGET_COL not in submission.schema.names:
        rep.fail(f"{SUBMISSION_TARGET_COL} missing")
        return rep
    y = submission[SUBMISSION_TARGET_COL].combine_chunks()
    if not (pa.types.is_integer(y.type) or pa.types.is_floating(y.type)):
        rep.fail(f"{SUBMISSION_TARGET_COL} is {y.type}, not numeric")
        return rep
    rep.n_null = int(pc.sum(pc.is_null(y, nan_is_null=True)).as_py() or 0)
    if rep.n_null:
        rep.fail(f"{rep.n_null:,} null/NaN values in {SUBMISSION_TARGET_COL}")
    vals = np.asarray(pc.drop_null(y).to_numpy(zero_copy_only=False), dtype=np.float64)
    if pa.types.is_floating(y.type):
        n_inf = int(np.isinf(vals).sum())
        if n_inf:
            rep.fail(f"{n_inf:,} infinite values in {SUBMISSION_TARGET_COL}")
        vals = vals[np.isfinite(vals)]
    if len(vals):
        rep.min_value, rep.max_value = float(vals.min()), float(vals.max())
        rep.n_negative = int((vals < 0).sum())
        if rep.n_negative:
            rep.fail(f"{rep.n_negative:,} negative values (min {rep.min_value})")
        rep.n_above_long = int((vals > LONG_SECONDS).sum())
        if rep.n_above_long:
            rep.warnings.append(f"{rep.n_above_long:,} rows above {LONG_SECONDS:,} s (max {rep.max_value:,.0f})")
        if len(vals) > 1 and np.unique(vals).size == 1:
            rep.fail(f"every prediction is {vals[0]}; the model is broken")

    _months_and_airports(rep, submission, template, ranking)
    return rep


def _months_and_airports(rep: FinalReport, submission: pa.Table, template: pa.Table, ranking: pa.Table | None) -> None:
    if ranking is None:
        rep.fail("no --ranking (final_ranking.parquet): per-month and per-airport checks did not run")
        return
    need = [SUBMISSION_ID_COL, "MVT_TIME_UTC_mvt", "ADEP_mvt"]
    missing = [c for c in need if c not in ranking.schema.names]
    if missing or SUBMISSION_ID_COL not in submission.schema.names:
        rep.fail(f"ranking file lacks {missing}")
        return
    rk = pl.from_arrow(ranking.select(need)).with_columns(
        pl.col(SUBMISSION_ID_COL).cast(pl.Float64),
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("_y"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("_m"),
    )
    if "PHASE_mvt" in ranking.schema.names:
        phase = pl.from_arrow(ranking.select(["PHASE_mvt"])).to_series()
        rk = rk.filter(phase == "DEP")
    if rk[SUBMISSION_ID_COL].is_duplicated().any():
        rep.fail("final_ranking has duplicate departure MVT_ID_mvt; month mapping is ambiguous")
        return
    other_label = pl.format("{}-{}", pl.col("_y"), pl.col("_m").cast(pl.String).str.zfill(2))
    label = pl.when(pl.col("_y").is_null()).then(pl.lit(None, pl.String))
    for (y, m), name in FINAL_MONTHS.items():
        label = label.when((pl.col("_y") == y) & (pl.col("_m") == m)).then(pl.lit(name))
    rk = rk.with_columns(label.otherwise(other_label).alias("_month")).select(SUBMISSION_ID_COL, "ADEP_mvt", "_month")

    sub = pl.from_arrow(submission.select([SUBMISSION_ID_COL, SUBMISSION_TARGET_COL])).with_columns(
        pl.col(SUBMISSION_ID_COL).cast(pl.Float64), pl.col(SUBMISSION_TARGET_COL).cast(pl.Float64)
    )
    tmpl = pl.from_arrow(template.select([SUBMISSION_ID_COL])).with_columns(pl.col(SUBMISSION_ID_COL).cast(pl.Float64))
    js = sub.join(rk, on=SUBMISSION_ID_COL, how="left")
    jt = tmpl.join(rk, on=SUBMISSION_ID_COL, how="left")
    n_unknown = js["_month"].null_count()
    if n_unknown:
        rep.fail(f"{n_unknown:,} ids are not departures in final_ranking")
    other = sorted(set(js["_month"].drop_nulls().unique().to_list()) - set(FINAL_MONTHS.values()))
    if other:
        rep.fail(f"rows outside Jan/Feb/Jun/Jul 2026: months {other}")
    tmpl_counts = dict(jt.group_by("_month").len().iter_rows())
    for month in [*FINAL_MONTHS.values(), *other]:
        part = js.filter(pl.col("_month") == month)[SUBMISSION_TARGET_COL].drop_nulls().drop_nans().to_numpy()
        entry = {
            "rows": int(js.filter(pl.col("_month") == month).height),
            "template_rows": int(tmpl_counts.get(month, 0)),
        }
        if len(part):
            entry["quantiles"] = {f"p{int(q * 100)}": float(np.quantile(part, q)) for q in QUANTILES}
            entry["mean"] = float(part.mean())
            entry["n_above_10800"] = int((part > LONG_SECONDS).sum())
        rep.months[month] = entry
        if month in FINAL_MONTHS.values() and entry["rows"] == 0:
            rep.fail(f"no rows for {month} 2026")
        if entry["rows"] != entry["template_rows"]:
            rep.fail(f"{month}: {entry['rows']:,} rows, template has {entry['template_rows']:,}")

    am = js.group_by("ADEP_mvt", "_month").len()
    for ap, month, n in sorted(am.iter_rows(), key=lambda r: (str(r[0]), str(r[1]))):
        rep.airport_month.setdefault(str(ap), {})[str(month)] = int(n)
    unexpected = sorted(set(rep.airport_month) - set(TARGET_AIRPORTS) - {"None"})
    if unexpected:
        rep.warnings.append(f"airports outside the 10 target airports: {unexpected}")
    for ap in TARGET_AIRPORTS:
        for month in FINAL_MONTHS.values():
            if rep.airport_month.get(ap, {}).get(month, 0) == 0:
                rep.warnings.append(f"{ap} has no rows in {month}")


def validate_final_files(
    file_path: Path,
    template_path: Path,
    ranking_path: Path | None,
    expect_md5: str | None = None,
    expected_rows: int | None = EXPECTED_FINAL_ROWS,
) -> FinalReport:
    with warnings.catch_warnings():  # the organisers' files carry an R (vctrs) extension type polars does not know
        warnings.simplefilter("ignore")
        sub = pq.read_table(file_path)
        tmpl = pq.read_table(template_path)
        cols = [SUBMISSION_ID_COL, "MVT_TIME_UTC_mvt", "ADEP_mvt", "PHASE_mvt"]
        if ranking_path is not None:
            names = pq.read_schema(ranking_path).names
            rank = pq.read_table(ranking_path, columns=[c for c in cols if c in names])
        else:
            rank = None
        rep = validate_final(
            sub, tmpl, rank, md5=file_md5(file_path), expect_md5=expect_md5, expected_rows=expected_rows
        )
    rep.file = str(file_path)
    return rep


def format_report(rep: FinalReport) -> str:
    out = [
        f"file: {rep.file}",
        f"md5:  {rep.md5}",
        f"rows: {rep.n_rows:,} (template {rep.template_rows:,})",
        f"schema:   {rep.schema}",
        f"template: {rep.template_schema}",
        (
            f"nulls: {rep.n_null:,}  negatives: {rep.n_negative:,}  min: {rep.min_value}  max: {rep.max_value}  "
            f"rows > {LONG_SECONDS:,} s: {rep.n_above_long:,}"
        ),
        "per month (rows / template rows / quantiles s):",
    ]
    for month, e in rep.months.items():
        q = e.get("quantiles", {})
        qs = "  ".join(f"{k}={v:,.0f}" for k, v in q.items())
        out.append(
            f"  {month}: {e['rows']:,} / {e['template_rows']:,}  mean={e.get('mean', float('nan')):,.1f}  {qs}  "
            f">10800: {e.get('n_above_10800', 0)}"
        )
    months = list(FINAL_MONTHS.values())
    out.append("per airport per month (rows): " + " ".join(f"{m:>7}" for m in months))
    for ap in sorted(rep.airport_month):
        row = rep.airport_month[ap]
        extra = {k: v for k, v in row.items() if k not in months}
        out.append(
            f"  {ap:<6}" + " ".join(f"{row.get(m, 0):>7,}" for m in months) + (f"  other {extra}" if extra else "")
        )
    for w in rep.warnings:
        out.append(f"WARNING: {w}")
    for e in rep.errors:
        out.append(f"ERROR: {e}")
    out.append(("PASS" if rep.ok else "FAIL") + f" ({rep.n_rows:,} rows, md5 {rep.md5})")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m prc.validate_final", description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--template", type=Path, required=True, help="final_submitting.parquet")
    ap.add_argument("--ranking", type=Path, help="final_ranking.parquet (months and airports)")
    ap.add_argument("--expect-md5", help="md5 the hand-off names; a mismatch is a hard failure")
    ap.add_argument("--json", type=Path, help="also write the full report as JSON here")
    args = ap.parse_args(argv)
    rep = validate_final_files(args.file, args.template, args.ranking, expect_md5=args.expect_md5)
    print(format_report(rep))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(rep), indent=1, default=str))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
