"""peaceful-babbage's ucmix (gated LIRF unmatched copy mixture) as a delta on the unmatched rows of a submission.

Fits only lane B (the target frame holds unmatched ranking rows only, so model A is never fitted) with the
peaceful-babbage code (run with PYTHONPATH=/home/user/pb-work/src from /home/user/pb-work). `_ucmix_blend` is
wrapped to record its input (the plain model-B prediction), so one fit gives both arms; delta = blended - plain.

    cd /home/user/pb-work && PYTHONPATH=src .venv/bin/python /home/user/prc-work/analysis/local_nmb_laneB.py <out.parquet>
"""
import sys
import time

import polars as pl
import yaml

import prc.train as T
from prc.pipeline import drop_hidden_candidates, train_config_from_spec

spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf_l2_opt_ucmix.yaml"))
cfg = train_config_from_spec(spec)
rec = {}
_orig = T._ucmix_blend


def _wrap(tr, u_tg, g, cfg_, cols_b, cat_cols, info):
    rec["ids"], rec["plain"] = u_tg["MVT_ID_mvt"], g.copy()
    out = _orig(tr, u_tg, g, cfg_, cols_b, cat_cols, info)
    rec["blend"] = out.copy()
    return out


T._ucmix_blend = _wrap
t0 = time.time()
from pathlib import Path

from prc.levers import attach_levers
from prc.pipeline import load_raw

f_tr, f_rk = pl.read_parquet("data/work/feat_train.parquet"), pl.read_parquet("data/work/feat_rank.parquet")
raw_tr, raw_rk, _ = load_raw(Path("data/raw"))
# same lever step as prc.pipeline.main (the recipe has fe_fwd_nm on)
f_tr, f_rk = attach_levers(f_tr, f_rk, raw_tr, raw_rk, l1=cfg.fe_plan_nm_taxi, l2=cfg.fe_fwd_nm or cfg.fe_fwd_nm_unmatched,
                           l3=cfg.fe_push_queue, l5=cfg.fe_tx_nm, l6=cfg.fe_retention)
del raw_tr, raw_rk
f_tr = drop_hidden_candidates(f_tr, cfg)
f_rk = drop_hidden_candidates(f_rk, cfg).filter(pl.col("mvt_minus_aobt3").is_null())
pred, info = T.predict_fold(f_tr, f_rk, cfg)
d = pl.DataFrame({"MVT_ID_mvt": rec["ids"], "plain": rec["plain"], "blend": rec["blend"]}).with_columns(
    (pl.col("blend") - pl.col("plain")).alias("delta")).join(f_rk.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt")
d.write_parquet(sys.argv[1])
print(f"{time.time() - t0:.0f}s rows {d.height} changed {int((d['delta'].abs() > 0.5).sum())} sum d^2/344841 "
      f"{float((d['delta'] ** 2).sum()) / 344841:.1f}", {k: v for k, v in info.items() if 'ucmix' in k or 'round' in k})
print(d.group_by("ADEP_mvt").agg(pl.len(), pl.col("delta").mean().round(1), pl.col("delta").abs().mean().round(1).alias("abs")).sort("ADEP_mvt"))
