"""Feature-addition verification at the baseline's frozen round counts.

Run arms sequentially. This never predicts ranking or calls the submission pipeline.
"""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
import json
import time
from pathlib import Path

import polars as pl
import yaml

import prc.train as training
from prc.train import TrainConfig, predict_fold

OUT = Path("outputs/codex_fe")
RES = Path("docs/codex_fe_results")


def main(arm):
    info = json.loads((RES / "baseline.json").read_text())
    selection = json.loads((RES / "selection.json").read_text())
    cols = selection["selected"] if arm == "selected" else (selection["joint_novel"] if arm == "all" else ["noise_0"])
    if not cols:
        print("No scalar candidates selected; no selected refit.", flush=True)
        return
    cfg = TrainConfig(**yaml.safe_load(Path("configs/codex_fe_screen.yaml").read_text()))
    cfg.name = "codex_fe_refit_" + arm
    cfg.inner_holdout_months = 0
    cfg.num_rounds = info["rounds_matched"]
    cfg.unmatched_num_rounds = info["rounds_unmatched"]
    f = pl.read_parquet("data/work/feat_train.parquet")
    cand = pl.read_parquet(OUT / "candidates_train.parquet", columns=["MVT_ID_mvt", *cols])
    f = f.join(cand, on="MVT_ID_mvt", maintain_order="left")
    del cand
    training.CATEGORICALS = list(training.CATEGORICALS) + [c for c in cols if f[c].dtype == pl.String]
    # Explicitly convert Boolean candidates: baseline numeric selector omits Boolean.
    f = f.with_columns([pl.col(c).cast(pl.Int8) for c in cols if f[c].dtype == pl.Boolean])
    tr = f.filter(pl.col("month") <= 8)
    te = f.filter(pl.col("month") >= 9)
    del f
    t = time.time()
    # Instrumentation only: preserve models and true column names for audit.
    original_matrix, original_train = training._matrix, training.lgb.train
    names = []
    importance = []

    def matrix(*args, **kwargs):
        nonlocal names
        values, names = original_matrix(*args, **kwargs)
        return values, names

    def fit(*args, **kwargs):
        model = original_train(*args, **kwargs)
        regime = "matched" if not importance else "unmatched"
        model.save_model(str(OUT / f"model_{arm}_{regime}.txt"))
        importance.append(
            {
                "regime": regime,
                "columns": names,
                "gain": model.feature_importance(importance_type="gain").tolist(),
                "splits": model.feature_importance(importance_type="split").tolist(),
            }
        )
        return model

    training._matrix, training.lgb.train = matrix, fit
    print("refit", arm, "columns", cols, "rounds", cfg.num_rounds, cfg.unmatched_num_rounds, flush=True)
    p, stats = predict_fold(tr, te, cfg)
    p.write_parquet(OUT / f"refit_{arm}.parquet")
    stats.update(seconds=time.time() - t, features=cols, config=cfg.__dict__)
    (RES / f"importance_{arm}.json").write_text(json.dumps(importance, indent=2))
    (RES / f"refit_{arm}.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["selected", "all", "noise"])
    main(p.parse_args().arm)
