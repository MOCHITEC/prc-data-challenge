import datetime as dt
import json

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from prc.config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL
from prc.features import build_departure_features
from prc.predict import predict_ranking, write_submission
from prc.submit import _today_count, submit
from prc.synth import write_synthetic_dataset
from prc.train import TrainConfig


def test_end_to_end_on_synthetic(tmp_path):
    paths = write_synthetic_dataset(tmp_path / "raw", months=[(2025, 1), (2025, 7)], deps_per_airport_day=6)
    train = pl.concat([pl.read_parquet(p) for n, p in paths.items() if n.startswith("training_")])
    rank = pl.read_parquet(paths["ranking.parquet"])
    ftr, frk = build_departure_features(train), build_departure_features(rank)
    pred, _ = predict_ranking(ftr, frk, TrainConfig(num_rounds=50, early_stopping=10))
    out = tmp_path / "sub" / "team_v1.parquet"
    sub = write_submission(pred, paths["submitting.parquet"], out)
    tmpl = pq.read_table(paths["submitting.parquet"])
    assert sub.schema.equals(tmpl.schema) and sub.num_rows == tmpl.num_rows
    log = tmp_path / "submissions.jsonl"
    e1 = submit(out, paths["submitting.parquet"], 1, "exp-x", {"rmse_full": 1.0}, "test", log=log, dry_run=True)
    assert e1["dry_run"] and log.exists()
    # dry runs must consume neither the daily quota nor a version number
    for v in (1, 1, 2, 3, 4, 5, 6, 7):
        submit(out, paths["submitting.parquet"], v, "exp-x", {}, "rehearsal", log=log, dry_run=True)
    assert _today_count(log)[0] == 0 and _today_count(log)[1] == set()
    assert json.loads(log.read_text().splitlines()[0])["version"] == 1


def test_daily_counter_uses_both_day_boundaries_and_ignores_dry_runs(tmp_path):
    """The quota's time zone is undocumented, so count on the UTC day and the Brussels day and take the larger."""
    log = tmp_path / "sub.jsonl"
    rows = [
        # 23:30Z on 9 Oct is already 01:30 on 10 Oct in Brussels (CEST)
        {"version": 1, "uploaded_at": "2026-10-09T23:30:00+00:00"},
        {"version": 2, "uploaded_at": "2026-10-10T00:10:00+00:00"},
        {"version": 3, "uploaded_at": "2026-10-10T05:00:00+00:00", "dry_run": True},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    now = dt.datetime(2026, 10, 10, 4, 0, tzinfo=dt.UTC)
    n, versions = _today_count(log, now=now)
    assert n == 2  # both count on the Brussels day; the dry run does not
    assert versions == {1, 2}


def test_gate_rejects_shuffled_predictions(tmp_path):
    """A submission whose values are shuffled has a valid id set, so only order-aware code catches it."""
    paths = write_synthetic_dataset(tmp_path / "raw", months=[(2025, 1)], deps_per_airport_day=3)
    tmpl = pq.read_table(paths["submitting.parquet"])
    ids = tmpl[SUBMISSION_ID_COL].to_pylist()
    pred = pl.DataFrame({"MVT_ID_mvt": ids, "pred": [float(i) for i in range(len(ids))]})
    good = write_submission(pred, paths["submitting.parquet"], tmp_path / "good.parquet")
    assert good[SUBMISSION_TARGET_COL].to_pylist() == [float(i) for i in range(len(ids))]
    # ids must come from the same frame as the values: a reversed prediction frame still lands correctly
    rev = pred.reverse()
    also = write_submission(rev, paths["submitting.parquet"], tmp_path / "rev.parquet")
    assert also[SUBMISSION_TARGET_COL].to_pylist() == good[SUBMISSION_TARGET_COL].to_pylist()


def test_integer_and_null_typed_templates(tmp_path):
    """docs/01 U13: the official template dtype is unconfirmed; int64 and null-typed columns must both work."""
    ids = pa.array([1, 2, 3], type=pa.int64())
    pred = pl.DataFrame({"MVT_ID_mvt": [1, 2, 3], "pred": [600.4, 700.6, 800.0]})
    for target, expect in (
        (pa.array([None] * 3, type=pa.int64()), [600, 701, 800]),
        (pa.array([None] * 3, type=pa.float64()), [600.4, 700.6, 800.0]),
        (pa.nulls(3, type=pa.null()), [600.4, 700.6, 800.0]),
    ):
        t = pa.table({SUBMISSION_ID_COL: ids, SUBMISSION_TARGET_COL: target})
        tp = tmp_path / f"tmpl_{target.type}.parquet"
        pq.write_table(t, tp)
        sub = write_submission(pred, tp, tmp_path / f"out_{target.type}.parquet")
        assert sub[SUBMISSION_TARGET_COL].to_pylist() == expect


def test_float_template_accepts_integer_typed_predictions(tmp_path):
    """The official files type the id as float64; a frame that has been through an int round trip is int64.

    The older dtype test used int64 on both sides, so the alignment it was meant to cover never ran."""
    t = pa.table(
        {
            SUBMISSION_ID_COL: pa.array([10.0, 20.0, 30.0], type=pa.float64()),
            SUBMISSION_TARGET_COL: pa.nulls(3, type=pa.float64()),
        }
    )
    tp = tmp_path / "float_tmpl.parquet"
    pq.write_table(t, tp)
    pred = pl.DataFrame({"MVT_ID_mvt": [30, 10, 20], "pred": [3.0, 1.0, 2.0]})  # int64 ids, deliberately unsorted
    assert pred["MVT_ID_mvt"].dtype != pl.Float64
    sub = write_submission(pred, tp, tmp_path / "float_out.parquet")
    assert sub[SUBMISSION_TARGET_COL].to_pylist() == [1.0, 2.0, 3.0], "values must follow their id, not their row"


def test_the_daily_counter_counts_versions_not_log_lines(tmp_path):
    """Every real upload writes a write-ahead 'attempted' entry and then the final one, so counting lines
    doubled the day's total: three uploads read as six and the fourth was refused."""
    import datetime as dt
    import json

    from prc.submit import _today_count

    log = tmp_path / "submissions.jsonl"
    now = dt.datetime(2026, 9, 15, 18, 0, tzinfo=dt.UTC)
    lines = []
    for v in (6, 7, 8):
        e = {"version": v, "uploaded_at": now.isoformat(), "dry_run": False}
        lines.append({**e, "status": "attempted"})
        lines.append({**e, "status": "uploaded"})
    lines.append({"version": 5, "uploaded_at": "2026-09-14T10:00:00+00:00", "dry_run": False})
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    n, versions = _today_count(log, now=now)
    assert n == 3, f"three uploads today, not {n}"
    assert versions == {5, 6, 7, 8}
