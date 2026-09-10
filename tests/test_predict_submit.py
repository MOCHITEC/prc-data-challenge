import json

import polars as pl
import pyarrow.parquet as pq

from prc.features import build_departure_features
from prc.predict import predict_ranking, write_submission
from prc.submit import submit
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
    try:
        submit(out, paths["submitting.parquet"], 1, "exp-x", {}, "dup", log=log, dry_run=True)
        raise AssertionError("duplicate version accepted")
    except ValueError:
        pass
    for v in range(2, 6):
        submit(out, paths["submitting.parquet"], v, "exp-x", {}, "n", log=log, dry_run=True)
    try:
        submit(out, paths["submitting.parquet"], 6, "exp-x", {}, "over", log=log, dry_run=True)
        raise AssertionError("daily limit not enforced")
    except RuntimeError:
        pass
    assert len(log.read_text().splitlines()) == 5
    assert json.loads(log.read_text().splitlines()[0])["version"] == 1
