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
from prc.features import revision_context_features, stand_arrival_features
from prc.train import TrainConfig, predict_fold

OUT = Path("outputs/codex_fe")
RES = Path("docs/codex_fe_results")


def main(arm, train_through=8):
    info = json.loads((RES / "baseline.json").read_text())
    selection = json.loads((RES / "selection.json").read_text())
    cols = selection["selected"] if arm == "selected" else (selection["joint_novel"] if arm == "all" else ["noise_0"])
    if arm == "gap":
        cols = ["stand_arrival_gap"]
    if arm == "baseline":
        cols = []
    tag = arm if train_through == 8 else f"{arm}_{train_through}m"
    cfg = TrainConfig(**yaml.safe_load(Path("configs/codex_fe_screen.yaml").read_text()))
    cfg.name = "codex_fe_refit_" + tag
    cfg.inner_holdout_months = 0
    cfg.num_rounds = info["rounds_matched"]
    cfg.unmatched_num_rounds = info["rounds_unmatched"]
    cfg.include_candidates = cols
    f = pl.read_parquet("data/work/feat_train.parquet")
    if arm in ["gap", "selected"]:
        arrivals = (
            pl.scan_parquet(sorted(Path("data/raw").glob("training_*.parquet")))
            .filter(pl.col("PHASE_mvt") == "ARR")
            .select("ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt")
            .collect()
        )
        cand = (
            stand_arrival_features(f, arrivals)
            .join(revision_context_features(f, f), on="MVT_ID_mvt", maintain_order="left")
            .select("MVT_ID_mvt", *cols)
        )
        del arrivals
        feature_source = "production feature helpers on read-only cached context"
    else:
        cand = pl.read_parquet(OUT / "candidates_train.parquet", columns=["MVT_ID_mvt", *cols])
        feature_source = "saved candidate sidecar"
    f = f.join(cand, on="MVT_ID_mvt", maintain_order="left")
    del cand
    training.CATEGORICALS = list(training.CATEGORICALS) + [c for c in cols if f[c].dtype == pl.String]
    # Explicitly convert Boolean candidates: baseline numeric selector omits Boolean.
    f = f.with_columns([pl.col(c).cast(pl.Int8) for c in cols if f[c].dtype == pl.Boolean])
    tr = f.filter(pl.col("month") <= train_through)
    te = f.filter(pl.col("month") > train_through)
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
        def progress(env):
            if (env.iteration + 1) % 100 == 0:
                print(f"{tag}: {env.iteration + 1} rounds; {time.time() - t:.1f}s elapsed", flush=True)

        kwargs["callbacks"] = [*kwargs.get("callbacks", []), progress]
        model = original_train(*args, **kwargs)
        regime = "matched" if not importance else "unmatched"
        model.save_model(str(OUT / f"model_{tag}_{regime}.txt"))
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
    p.write_parquet(OUT / f"refit_{tag}.parquet")
    stats.update(
        seconds=time.time() - t,
        features=cols,
        config=cfg.__dict__,
        train_through=train_through,
        feature_source=feature_source,
    )
    (RES / f"importance_{tag}.json").write_text(json.dumps(importance, indent=2))
    (RES / f"refit_{tag}.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["selected", "all", "noise", "gap", "baseline"])
    p.add_argument("--train-through", type=int, choices=[8, 10], default=8)
    args = p.parse_args()
    main(args.arm, args.train_through)
