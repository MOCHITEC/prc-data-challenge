"""Where is the error on 'genuine' rows (BLOCK is a real off-block, not a copy / fill / day shift)?

Matched-lane LightGBM (y - proxy) trained on non-scored months, scored on Jan+Jul 2025. Genuine = BLOCK not within
60 s of SCHED/AOBT_3/EOBT_1/LOBT/IOBT and y < 20 h. Reports SSE share and RMSE by candidate groups."""

import numpy as np
import polars as pl
import lightgbm as lgb

f = pl.read_parquet("data/work/feat_train.parquet").filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
f = f.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
m = f.filter(pl.col("mvt_minus_aobt3").is_not_null())
num = [c for c, d in m.schema.items() if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Int16, pl.UInt32, pl.UInt8)
       and c not in ("TAXITIME_SEC_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "month", "mo")]
tr, te = m.filter(~pl.col("mo").is_in([1, 7])), m.filter(pl.col("mo").is_in([1, 7]))
X = lambda d: d.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy()  # noqa: E731
params = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
bst = lgb.train(params, lgb.Dataset(X(tr), (tr["TAXITIME_SEC_mvt"] - tr["mvt_minus_aobt3"]).to_numpy()), 600)
te = te.with_columns(pl.Series("p", te["mvt_minus_aobt3"].to_numpy() + bst.predict(X(te))))
y = pl.col("TAXITIME_SEC_mvt")
copy = pl.lit(False)
for c in ("mvt_minus_sched", "mvt_minus_aobt3", "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt"):
    copy = copy | ((y - pl.col(c)).abs() <= 60).fill_null(False)
te = te.with_columns(
    copy.alias("copy"), (y - pl.col("p")).alias("e"),
    (pl.col("aobt3_minus_sched") / 60).alias("delay_min"),
    (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("block_minus_aobt3"),
)
g = te.filter(~pl.col("copy") & (y < 72000))
tot = float((te["e"] ** 2).sum())
print(f"all matched Jan+Jul: n={te.height} RMSE={np.sqrt((te['e']**2).mean()):.1f}")
print(f"genuine: n={g.height} ({g.height/te.height:.1%}) RMSE={np.sqrt((g['e']**2).mean()):.1f} share of SSE {float((g['e']**2).sum())/tot:.1%}")
G = float((g["e"] ** 2).sum())


def show(name, expr, top=12):
    t = (g.with_columns(expr.alias("grp")).group_by("grp")
         .agg(pl.len().alias("n"), (pl.col("e") ** 2).sum().alias("sse"), pl.col("e").mean().alias("bias"),
              y.median().alias("y_med"))
         .with_columns((pl.col("sse") / G).alias("sse_share"), (pl.col("n") / g.height).alias("row_share"),
                       (pl.col("sse") / pl.col("n")).sqrt().alias("rmse"))
         .with_columns((pl.col("sse_share") / pl.col("row_share")).alias("lift"))
         .sort("sse", descending=True).head(top)
         .select("grp", "n", "row_share", "sse_share", "lift", "rmse", "bias", "y_med"))
    print(f"\n== by {name}"); print(t)


pl.Config.set_tbl_rows(20); pl.Config.set_float_precision(3); pl.Config.set_tbl_width_chars(160)
show("airport", pl.col("ADEP_mvt"))
show("|error| band", pl.col("e").abs().cut([120, 300, 600, 1200, 3600], labels=["<2m", "2-5m", "5-10m", "10-20m", "20-60m", ">60m"]))
show("BLOCK - AOBT_3 (how far the airport clock is from NM)", pl.col("block_minus_aobt3").cut([-1800, -600, -120, 120, 600, 1800], labels=["<-30m", "-30..-10m", "-10..-2m", "+-2m", "2..10m", "10..30m", ">30m"]))
show("taxi length y", y.cut([600, 900, 1200, 1800, 2700, 3600, 7200], labels=["<10m", "10-15", "15-20", "20-30", "30-45", "45-60", "1-2h", ">2h"]))
show("NM delay (AOBT3 - SCHED)", pl.col("delay_min").cut([-10, 5, 15, 30, 60, 180], labels=["early", "on time", "5-15", "15-30", "30-60", "1-3h", ">3h"]))
show("month x deicing gate", pl.format("{}-{}", pl.col("mo"), pl.col("deicing_gate").fill_null(-1)))
show("local hour", pl.col("hour_local"))
show("wake category", pl.col("WK_TBL_CAT_flt"))
show("market segment", pl.col("MARKET_SEGMENT_flt"))
show("airport x runway (top)", pl.format("{}-{}", pl.col("ADEP_mvt"), pl.col("RUNWAY_mvt")), top=15)
show("day (airport-date, top)", pl.format("{}-{}", pl.col("ADEP_mvt"), pl.col("MVT_TIME_UTC_mvt").dt.date()), top=15)
