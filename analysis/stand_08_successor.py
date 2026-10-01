"""Tightening the stand-occupancy bound with the successor arrival's taxi-in.

If the next aircraft assigned to our stand landed *before* we pushed back, it may have had to wait for the stand,
so its in-block A follows our off-block by only the push-back clearance time. Its taxi-in (A - landing) is
ranking-visible (ARR rows keep BLOCK and MVT). We measure, for departures whose latest same-stand in-block
before take-off is < 30 min old (the bound candidate), how the slack  y - (MVT - A)  depends on
  * the successor's taxi-in excess over the stand's median taxi-in (fitted on 2025 arrivals)
  * whether the successor landed before our take-off - (MVT - landing)
Writes SCRATCH/stand_succ_feats.parquet (succ_taxiin, succ_taxiin_excess, succ_land_before_tko_s).
"""

from __future__ import annotations

import polars as pl

from stand_common import SCRATCH, load_all

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)


def main() -> None:
    df = load_all()
    arr = df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("SKEY").is_not_null() & pl.col("t_blk").is_not_null())
    arr = arr.with_columns((pl.col("t_blk") - pl.col("t_mvt")).alias("taxi_in"))
    med = arr.filter(pl.col("split") == "train").group_by("SKEY").agg(pl.col("taxi_in").median().alias("ti_med"))
    arr = arr.join(med, on="SKEY", how="left").with_columns((pl.col("taxi_in") - pl.col("ti_med")).alias("ti_exc"))
    dep = df.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("SKEY").is_not_null())
    a = arr.select(["SKEY", pl.col("t_blk").alias("ta"), pl.col("t_mvt").alias("tl"), "taxi_in", "ti_exc"]).sort("ta")
    j = dep.select(["MVT_ID_mvt", "SKEY", "APT", "t_mvt", "t_aobt", "y", "split"]).sort("t_mvt").join_asof(
        a, left_on="t_mvt", right_on="ta", by="SKEY", strategy="backward")
    j = j.with_columns(
        (pl.col("t_mvt") - pl.col("ta")).alias("g"),
        (pl.col("t_mvt") - pl.col("tl")).alias("succ_land_before_tko_s"),
    )
    out = j.select(["MVT_ID_mvt",
                    pl.when(pl.col("g") < 3600).then(pl.col("taxi_in")).alias("succ_taxiin"),
                    pl.when(pl.col("g") < 3600).then(pl.col("ti_exc")).alias("succ_taxiin_excess"),
                    pl.when(pl.col("g") < 3600).then(pl.col("succ_land_before_tko_s")).alias("succ_land_before_tko_s")])
    out.write_parquet(f"{SCRATCH}/stand_succ_feats.parquet")

    tr = j.filter((pl.col("split") == "train") & (pl.col("g") < 1500) & (pl.col("y") >= 30) & (pl.col("y") <= 7200))
    tr = tr.with_columns((pl.col("y") - pl.col("g")).alias("slack"), (pl.col("y") - (pl.col("t_mvt") - pl.col("t_aobt"))).alias("resid"))
    print("## slack y - (MVT - A) by successor taxi-in excess (s), deps with same-stand in-block < 25 min before take-off")
    print(tr.group_by(pl.col("ti_exc").cut([-120, 0, 120, 300, 600, 1200]).alias("ti_exc_bucket")).agg(
        pl.len(), pl.col("slack").median().alias("slack_med"), pl.col("slack").quantile(0.25).alias("slack_p25"),
        (pl.col("slack") < 0).mean().alias("viol"), pl.col("resid").mean().alias("resid_mean")).sort("ti_exc_bucket"))
    print("## slack by successor landing time relative to our take-off (MVT - landing, s)")
    print(tr.group_by(pl.col("succ_land_before_tko_s").cut([0, 300, 600, 900, 1200, 1800]).alias("land_bucket")).agg(
        pl.len(), pl.col("slack").median().alias("slack_med"), (pl.col("slack") < 0).mean().alias("viol"),
        pl.col("y").mean().alias("y_mean"), pl.col("resid").mean().alias("resid_mean")).sort("land_bucket"))
    # best simple estimator of y in the bound region: g + c(ti_exc)
    tr = tr.with_columns(pl.col("ti_exc").cut([-120, 0, 120, 300, 600, 1200]).alias("b"))
    c = tr.group_by("b").agg(pl.col("slack").median().alias("c"))
    t2 = tr.join(c, on="b")
    import numpy as np
    y, g, cc = t2["y"].to_numpy(), t2["g"].to_numpy(), t2["c"].to_numpy()
    x = (t2["t_mvt"] - t2["t_aobt"]).to_numpy()
    ok = np.isfinite(x)
    print("RMSE in bound region: y~g+median_slack", float(np.sqrt(np.mean((y - g - np.median(y - g)) ** 2))),
          " y~g+c(ti_exc)", float(np.sqrt(np.mean((y - g - cc) ** 2))),
          " y~(MVT-AOBT3)+mean_resid", float(np.sqrt(np.mean((y[ok] - x[ok] - np.mean(y[ok] - x[ok])) ** 2))))


if __name__ == "__main__":
    main()
