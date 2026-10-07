"""Per-member effect of the fl_/seq_ train-test fix (union build) on the matched lane, with the peaceful-babbage code.

One fit of <config> (n_seeds overridden), predicting the original and the union-fixed ranking rows of the matched lane
together, as `local_flrate_fix.py` does, but with the lever columns the recipe asks for (attach_levers as in
prc.pipeline.main). Run from /home/user/pb-work with PYTHONPATH=src.

    PYTHONPATH=src .venv/bin/python /home/user/prc-work/analysis/local_flfix_member.py <config.yaml> <n_seeds> <out.parquet>
"""
import os
import sys
import time
from pathlib import Path

import polars as pl
import yaml

from prc.levers import attach_levers
from prc.pipeline import drop_hidden_candidates, load_raw, train_config_from_spec
from prc.train import predict_fold

cfg_path, n_seeds, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = yaml.safe_load(open(cfg_path))
spec["n_seeds"] = n_seeds
cfg = train_config_from_spec(spec)
t0 = time.time()
f_tr, f_rk = pl.read_parquet("data/work/feat_train.parquet"), pl.read_parquet("data/work/feat_rank.parquet")
if cfg.fe_plan_nm_taxi or cfg.fe_fwd_nm or cfg.fe_fwd_nm_unmatched or cfg.fe_push_queue or cfg.fe_tx_nm or cfg.fe_retention:
    raw_tr, raw_rk, _ = load_raw(Path("data/raw"))
    f_tr, f_rk = attach_levers(f_tr, f_rk, raw_tr, raw_rk, l1=cfg.fe_plan_nm_taxi, l2=cfg.fe_fwd_nm or cfg.fe_fwd_nm_unmatched,
                               l3=cfg.fe_push_queue, l5=cfg.fe_tx_nm, l6=cfg.fe_retention)
    del raw_tr, raw_rk
f_tr = drop_hidden_candidates(f_tr, cfg)
f_rk = drop_hidden_candidates(f_rk, cfg).filter(pl.col("mvt_minus_aobt3").is_not_null())
fix = pl.read_parquet(os.path.join(os.environ["SCRATCH"], "agents/audit/feat_rank_extra_union.parquet"))
fcols = [c for c in fix.columns if c != "MVT_ID_mvt" and c in f_rk.columns]
f_fx = (f_rk.drop(fcols).join(fix.select("MVT_ID_mvt", *fcols), on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns([pl.col(c).cast(f_rk[c].dtype) for c in fcols] + [(-pl.col("MVT_ID_mvt")).alias("MVT_ID_mvt")]))
tg = pl.concat([f_rk, f_fx.select(f_rk.columns)])
print(f"{spec['name']}: swapped {len(fcols)} cols, train {f_tr.height:,} target {tg.height:,} ({time.time() - t0:.0f}s)", flush=True)
pred, info = predict_fold(f_tr, tg, cfg)
p = pred.filter(pl.col("MVT_ID_mvt") > 0).rename({"pred": "pred_orig"}).join(
    pred.filter(pl.col("MVT_ID_mvt") < 0).with_columns(-pl.col("MVT_ID_mvt")).rename({"pred": "pred_fix"}), on="MVT_ID_mvt")
p = p.join(f_rk.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt")
p.write_parquet(out)
d = p["pred_fix"] - p["pred_orig"]
print(f"done {time.time() - t0:.0f}s rows {p.height} mean {d.mean():+.2f} mean|d| {d.abs().mean():.2f} sum d^2/344841 {float((d ** 2).sum()) / 344841:.1f}")
