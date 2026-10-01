"""P10 Stage 1: one fold, a deterministic subsample of the TRAINING rows, the full validation month(s).

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/p10_slice.py \
        --config configs/p10_cb_ne.yaml --fold forward --frac 0.2 --tag cb_ne_f20

Plumbing validation, a memory probe and a smell test - not a decision (P10 §5 Stage 1). The subsample is a
pure function of MVT_ID_mvt (`prc.train._row_uniform`, the murmur3 finaliser the P11 control uses), so every
arm run at the same --frac sees exactly the same training rows. The validation side is the fold's whole
validation set, so the RMSE is on the same rows the full-data arm will be scored on.

Goes through `predict_fold` - the same function `run_cv` calls - so the fitter, encodings, routing, target
and assembly are exactly the production path. Writes a JSON line to results/p10-runs/slice.jsonl and the
predictions to results/p10-runs/slice_<tag>.parquet. Never touches experiments/ or outputs/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from prc.cv import metrics, month_block_folds, year_month_key
from prc.pipeline import train_config_from_spec
from prc.train import LABEL, PROXY, _row_uniform, predict_fold

OUT = Path("docs/competitor_research_2026-09-21/results/p10-runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", default="forward")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--set", default="{}", help="JSON of TrainConfig overrides applied on top of the recipe")
    a = ap.parse_args()
    spec = {**yaml.safe_load(a.config.read_text()), **json.loads(a.set)}
    cfg = train_config_from_spec(spec)
    t0 = time.time()
    feats = pl.read_parquet(a.work_dir / "feat_train.parquet")
    months = year_month_key(feats)
    fold = next(f for f in month_block_folds(months.unique().to_list()) if f.name == a.fold)
    tr = feats.filter(months.is_in(list(fold.train_months)))
    va = feats.filter(months.is_in(list(fold.valid_months)))
    del feats, months
    keep = _row_uniform(tr["MVT_ID_mvt"].to_numpy(), 20_260_928) < a.frac
    tr = tr.filter(pl.Series("_k", keep))
    t_load = time.time() - t0
    p, info = predict_fold(tr, va, cfg)
    t_fit = time.time() - t0 - t_load
    j = va.select(["MVT_ID_mvt", LABEL, PROXY]).join(p, on="MVT_ID_mvt", how="inner")
    y, pv = j[LABEL].to_numpy().astype(float), j["pred"].to_numpy().astype(float)
    mt = j[PROXY].is_not_null().to_numpy()
    rec = {
        "tag": a.tag,
        "config": str(a.config),
        "overrides": json.loads(a.set),
        "exp_hash": cfg.hash(),
        "fold": a.fold,
        "frac": a.frac,
        "train_rows": tr.height,
        "valid_rows": va.height,
        "all": metrics(y, pv),
        "lane_a": metrics(y[mt], pv[mt]),
        "lane_b": metrics(y[~mt], pv[~mt]),
        "info": info,
        "load_seconds": round(t_load, 1),
        "fit_seconds": round(t_fit, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "slice.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    p.write_parquet(OUT / f"slice_{a.tag}.parquet")
    print(json.dumps({k: rec[k] for k in ("tag", "train_rows", "fit_seconds")}, default=str))
    print("rmse", rec["all"]["rmse_full"], "laneA", rec["lane_a"]["rmse_full"], "laneB", rec["lane_b"]["rmse_full"])
    print("rounds", info.get("rounds_matched"), info.get("rounds_unmatched"))
    return 0 if np.isfinite(pv).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
