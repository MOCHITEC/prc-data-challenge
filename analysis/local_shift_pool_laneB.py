"""Shift worker, task 3d: lane-B check of the fl_nm_rate_other pooling cost with v40's LIRF reparameterisation.

Same emulation as local_shift_pool_ablate.py (Jan+Jul 2025 lane-B rows, fl_nm_rate_other pooled over the 2-month
frame vs the 12-month frame), but model B's target is y - mvt_minus_sched at LIRF (unmatched_sched_reparam, as in
main_flseq_osm_nolirf) and five seeds are averaged, to see whether the LIRF lane-B damage survives that recipe.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_shift_pool_laneB.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from local_shift_pool import catcode, mat  # noqa: E402
from prc.features import CATEGORICALS  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import unmatched_columns, visible_numeric_columns  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "shift")
spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf.yaml"))
for k in ("name", "use_weather", "wx_codes", "wind_dir", "gust_repair", "coverage_gate", "coverage_allow", "note"):
    spec.pop(k, None)
cfg = train_config_from_spec(spec)
head = pl.read_parquet("data/work_flseq/feat_rank.parquet", n_rows=10)
num = [c for c in visible_numeric_columns(head, cfg) if not c.startswith("noise_control")]
cols = unmatched_columns(num)
cats = [c for c in CATEGORICALS if c in head.columns]
need = list(dict.fromkeys(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched", "mvt_minus_aobt3"] + cols + cats))
d = (pl.scan_parquet("data/work_flseq/feat_train.parquet").select(need)
     .filter(pl.col("mvt_minus_aobt3").is_null()).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect())
off = pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.col("mvt_minus_sched").fill_null(0)).otherwise(0).cast(pl.Float64)
d = d.with_columns(off.alias("off"))
fit, v = d.filter(~pl.col("mon").is_in([1, 7])), d.filter(pl.col("mon").is_in([1, 7]))
ex = pl.read_parquet(os.path.join(OUT, "pool_extra_2m.parquet"), columns=["MVT_ID_mvt", "fl_nm_rate_other"])
v2 = v.drop("fl_nm_rate_other").join(ex, on="MVT_ID_mvt", how="left").select(v.columns)
Cf, C1, C2 = catcode([fit, v, v2], cats)
Xf, X1, X2 = mat(fit, cols, Cf), mat(v, cols, C1), mat(v2, cols, C2)
yf = fit["TAXITIME_SEC_mvt"].to_numpy() - fit["off"].to_numpy()
p1 = np.zeros(v.height)
p2 = np.zeros(v.height)
for s in range(5):
    p = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=s)
    b = lgb.train(p, lgb.Dataset(Xf, yf, categorical_feature=list(range(len(cols), len(cols) + len(cats)))), 1000)
    p1 += b.predict(X1) / 5
    p2 += b.predict(X2) / 5
o = v["off"].to_numpy()
R = pl.DataFrame({"ADEP_mvt": v["ADEP_mvt"], "mon": v["mon"], "y": v["TAXITIME_SEC_mvt"].cast(pl.Float64), "p1": p1 + o, "p2": p2 + o})
N_all = 344419  # 2025 Jan+Jul departures (both lanes), so the numbers are MSE units on that set
R = R.with_columns(((pl.col("y") - pl.col("p2")) ** 2 - (pl.col("y") - pl.col("p1")) ** 2).alias("dd"))
with pl.Config(tbl_rows=30, float_precision=1):
    print(R.group_by("ADEP_mvt").agg(pl.len(), ((pl.col("y") - pl.col("p1")) ** 2).mean().sqrt().alias("rmse12"),
                                     (pl.col("dd").sum() / N_all).alias("dMSE_N"), (pl.col("p2") - pl.col("p1")).mean().alias("shift"),
                                     (pl.col("dd").filter(pl.col("mon") == 1).sum() / N_all).alias("jan"),
                                     (pl.col("dd").filter(pl.col("mon") == 7).sum() / N_all).alias("jul")).sort("ADEP_mvt"))
print("lane B total dMSE (MSE units on Jan+Jul):", R["dd"].sum() / N_all)
