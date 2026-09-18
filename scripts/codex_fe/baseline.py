"""Fresh common-model predictions; no use of the mixed historical OOF."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import hashlib
import json
import time
from pathlib import Path

import polars as pl
import yaml

from prc.train import TrainConfig, predict_fold

OUT = Path("outputs/codex_fe")
RES = Path("docs/codex_fe_results")
if __name__ == "__main__":
    t = time.time()
    cfg = TrainConfig(**yaml.safe_load(Path("configs/codex_fe_screen.yaml").read_text()))
    f = pl.read_parquet("data/work/feat_train.parquet")
    train = f.filter(pl.col("month") <= 8)
    target = f.filter(pl.col("month") >= 9)
    del f
    print("training", train.height, "predicting", target.height, flush=True)
    p, info = predict_fold(train, target, cfg)
    p.write_parquet(OUT / "baseline.parquet")
    info.update(
        seconds=time.time() - t,
        config=cfg.__dict__,
        feature_source="data/work/feat_train.parquet",
        feature_source_sha256=hashlib.sha256(Path("data/work/feat_meta.json").read_bytes()).hexdigest(),
    )
    (RES / "baseline.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)
