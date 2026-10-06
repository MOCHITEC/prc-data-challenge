"""Shift worker, task 4: what union pooling of fl_nm_rate_other would change on the 2026 lane-B rows (proxy model).

Lane-B proxy (main_flseq_osm_nolirf's unmatched columns, LIRF sched reparam, 5 seeds) trained on all twelve 2025
months with the production (12-month-pooled) features. Predicts the 2026 lane-B rows with
  (a) fl_nm_rate_other as in feat_rank (pooled over the Jan+Jul 2026 frame - what v40 saw), and
  (b) fl_nm_rate_other pooled over 2025+2026 departures (the candidate fix).
Reports per airport mean(b - a), mean |b - a|, and how v40's submitted values relate to proxy (a).

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_shift_pool_2026.py
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
need = list(dict.fromkeys(["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched", "mvt_minus_aobt3", "FLIGHT_mvt",
                           "FLIGHT_ID_mvt"] + cols + cats))
tr = pl.read_parquet("data/work_flseq/feat_train.parquet", columns=need)
rk = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=need)
u = pl.concat([tr.select("MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt"),
               rk.select("MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt")])
g = ["ADEP_mvt", "FLIGHT_mvt"]
n_other = pl.len().over(g) - 1
m = pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Float64)
u = u.with_columns(pl.when(pl.col("FLIGHT_mvt").is_not_null() & (n_other > 0))
                   .then((m.sum().over(g) - m) / n_other).otherwise(None).cast(pl.Float32).alias("fl_union"))
off = pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.col("mvt_minus_sched").fill_null(0)).otherwise(0).cast(pl.Float64)
fit = tr.filter(pl.col("mvt_minus_aobt3").is_null()).with_columns(off.alias("off"))
va = rk.filter(pl.col("mvt_minus_aobt3").is_null()).with_columns(off.alias("off"))
del tr, rk
vb = va.drop("fl_nm_rate_other").join(u.select("MVT_ID_mvt", pl.col("fl_union").alias("fl_nm_rate_other")),
                                      on="MVT_ID_mvt", how="left").select(va.columns)
Cf, Ca, Cb = catcode([fit, va, vb], cats)
Xf, Xa, Xb = mat(fit, cols, Cf), mat(va, cols, Ca), mat(vb, cols, Cb)
yf = fit["TAXITIME_SEC_mvt"].to_numpy() - fit["off"].to_numpy()
pa = np.zeros(va.height)
pb = np.zeros(va.height)
for s in range(5):
    p = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=s)
    b = lgb.train(p, lgb.Dataset(Xf, yf, categorical_feature=list(range(len(cols), len(cols) + len(cats)))), 1000)
    pa += b.predict(Xa) / 5
    pb += b.predict(Xb) / 5
o = va["off"].to_numpy()
v40 = pl.read_parquet("/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet").rename({"TAXITIME_SEC_mvt": "v40"})
R = (pl.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"], "ADEP_mvt": va["ADEP_mvt"], "pa": pa + o, "pb": pb + o,
                   "fl_a": va["fl_nm_rate_other"], "fl_b": vb["fl_nm_rate_other"]})
     .join(v40, on="MVT_ID_mvt", how="left").with_columns(pl.col("v40").cast(pl.Float64)))
R.write_parquet(os.path.join(OUT, "pool_2026_laneB.parquet"))
with pl.Config(tbl_rows=30, float_precision=1, tbl_width_chars=200):
    print(R.group_by("ADEP_mvt").agg(
        pl.len(), pl.col("fl_a").mean().alias("fl_now"), pl.col("fl_b").mean().alias("fl_union"),
        (pl.col("pb") - pl.col("pa")).mean().alias("mean_b_minus_a"), (pl.col("pb") - pl.col("pa")).abs().mean().alias("mean_abs"),
        pl.corr("v40", "pa").alias("corr_v40_pa"), (pl.col("v40") - pl.col("pa")).mean().alias("v40_minus_pa"),
        (((pl.col("pb") - pl.col("pa")) ** 2).sum() / 344841).alias("sum_sq_change_N")).sort("ADEP_mvt"))
