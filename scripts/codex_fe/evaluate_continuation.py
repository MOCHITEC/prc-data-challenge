"""Exploratory gap and training-size sensitivity; separate from locked bundle tests."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
import json
from pathlib import Path

import numpy as np
import polars as pl
from evaluate_refits import details
from screen import load

out = Path("outputs/codex_fe")
f = load().filter(pl.col("month") >= 11)
results = {}


def prediction(tag):
    p = pl.read_parquet(out / f"refit_{tag}.parquet")
    assert p["MVT_ID_mvt"].n_unique() == p.height
    joined = f.select("MVT_ID_mvt").join(p, on="MVT_ID_mvt", how="left", maintain_order="left", validate="1:1")
    assert joined["pred"].null_count() == 0
    return joined["pred"].to_numpy()


for candidate, control in [
    ("gap", "baseline"),
    ("gap", "noise"),
    ("selected", "gap"),
    ("all", "selected"),
    ("gap_10m", "baseline_10m"),
    ("baseline_10m", "baseline"),
    ("gap_10m", "gap"),
    ("selected_10m", "baseline_10m"),
    ("selected_10m", "selected"),
]:
    if not (out / f"refit_{candidate}.parquet").exists():
        continue
    if control != "baseline" and not (out / f"refit_{control}.parquet").exists():
        continue
    b = f["pred"].to_numpy() if control == "baseline" else prediction(control)
    p = prediction(candidate)
    assert np.isfinite(p).all()
    r = details(f.with_columns(pl.Series("pred", b)), p, alpha=0.05)
    results[candidate + "_vs_" + control] = r
    print(candidate, control, r["delta"], r["ci"], r["week_ci"], flush=True)
Path("docs/codex_fe_results/continuation_evaluation.json").write_text(json.dumps(results, indent=2))
