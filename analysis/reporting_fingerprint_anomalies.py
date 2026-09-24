"""Anomalous records and ranking-visible predicates for the extreme tail (report §6)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
from reporting_fingerprint_flightfmt import fmt_expr
import numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
df = load_all()
dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(fmt_expr(),
    (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("m_s"),
    pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null"))
# duplicates: same airport, same flight number, same scheduled time
dup = dep.group_by("split", "apt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt").agg(pl.len().alias("n"), pl.col("nm_null").sum().alias("n_nm_null"),
        pl.col("TAXITIME_SEC_mvt").max().alias("ymax"), pl.col("TAXITIME_SEC_mvt").min().alias("ymin"),
        ((pl.col("MVT_TIME_UTC_mvt").max() - pl.col("MVT_TIME_UTC_mvt").min()).dt.total_seconds()).alias("mvt_span")).filter((pl.col("n") > 1) & pl.col("FLIGHT_mvt").is_not_null())
print("duplicate (apt, flight, sched) DEP groups:", dup.group_by("split").agg(pl.len(), pl.col("n").sum(), pl.col("n_nm_null").sum()))
print(dup.filter(pl.col("split") == "train").select("apt", "FLIGHT_mvt", "n", "n_nm_null", "ymin", "ymax", "mvt_span").sort("ymax", descending=True).head(12))
tr = dep.filter(pl.col("split") == "train")
dupkeys = dup.select("split", "apt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt").with_columns(pl.lit(True).alias("is_dup"))
tr = tr.join(dupkeys, on=["split", "apt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], how="left").with_columns(pl.col("is_dup").fill_null(False))
print("dup rows y stats:", tr.group_by("is_dup", "nm_null").agg(pl.len(), pl.col("TAXITIME_SEC_mvt").mean().alias("ymean"), pl.col("TAXITIME_SEC_mvt").median().alias("ymed"), pl.col("TAXITIME_SEC_mvt").std().alias("ysd"), (pl.col("clock_src") == "sched").mean().alias("sched"), (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("gt1h")).sort("is_dup", "nm_null"))
print("dup rows by apt:", tr.filter(pl.col("is_dup")).group_by("apt").agg(pl.len(), pl.col("nm_null").sum()).sort("len", descending=True).to_dicts())
dd = tr.filter(pl.col("is_dup") & pl.col("nm_null")).with_columns(((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).alias("b_s"))
print("unmatched duplicates: BLOCK - SCHED quantiles", dd["b_s"].quantile(0.1), dd["b_s"].median(), dd["b_s"].quantile(0.9), " MVT-SCHED median", dd["m_s"].median())
# y of unmatched duplicate vs its matched twin's (MVT - AOBT3)
tw = tr.filter(pl.col("is_dup")).group_by("apt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt").agg(
    pl.col("TAXITIME_SEC_mvt").filter(pl.col("nm_null")).first().alias("y_unm"), pl.col("TAXITIME_SEC_mvt").filter(~pl.col("nm_null")).first().alias("y_m"),
    ((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds()).filter(~pl.col("nm_null")).first().alias("twin_mvt_aobt3"),
    pl.col("m_s").filter(pl.col("nm_null")).first().alias("unm_m_s"),
    ((pl.col("BLOCK_TIME_UTC_mvt").filter(pl.col("nm_null")).first() - pl.col("BLOCK_TIME_UTC_mvt").filter(~pl.col("nm_null")).first()).dt.total_seconds()).alias("blk_diff")).drop_nulls(["y_unm", "y_m"])
print("twins:", tw.height); print(tw.select("y_unm", "y_m", "twin_mvt_aobt3", "unm_m_s", "blk_diff").describe())
print("corr(y_unm, unm_m_s)", tw.select(pl.corr("y_unm", "unm_m_s")).item(), " share |y_unm - unm_m_s|<=60:", (tw["y_unm"] - tw["unm_m_s"]).abs().le(60).mean())
# negative y
neg = tr.filter(pl.col("TAXITIME_SEC_mvt") < 0)
print("negative y:", neg.height, neg.group_by("apt").len().sort("len", descending=True).to_dicts())
print(neg.select("apt", "clock_src", "nm_null", "TAXITIME_SEC_mvt", "m_s").describe())
# tail predicates
y = pl.col("TAXITIME_SEC_mvt")
preds = {
    "all": pl.lit(True),
    "LIRF": pl.col("apt") == "LIRF",
    "nm_null": pl.col("nm_null"),
    "nm_null & fmt in {dup,null}": pl.col("nm_null") & pl.col("fl_fmt").is_in(["dup_prefix", "null"]),
    "LIRF & nm_null": (pl.col("apt") == "LIRF") & pl.col("nm_null"),
    "LIRF & nm_null & m_s>3h": (pl.col("apt") == "LIRF") & pl.col("nm_null") & (pl.col("m_s") > 3 * 3600),
    "nm_null & m_s>3h": pl.col("nm_null") & (pl.col("m_s") > 3 * 3600),
    "nm_null & fmt dup/null & m_s>3h": pl.col("nm_null") & pl.col("fl_fmt").is_in(["dup_prefix", "null"]) & (pl.col("m_s") > 3 * 3600),
    "m_s>3h": pl.col("m_s") > 3 * 3600,
    "!nm_null & m_s>3h": (~pl.col("nm_null")) & (pl.col("m_s") > 3 * 3600),
}
tot3 = tr.filter(y > 3 * 3600).height
tot1 = tr.filter(y > 3600).height
sse_tot = tr.select(((y - y.mean()) ** 2).sum()).item()
rows = []
for k, p in preds.items():
    s = tr.filter(p)
    r = s.filter(pl.col("split") == "train")
    rank_n = dep.filter((pl.col("split") == "rank") & p).height
    rows.append(dict(pred=k, n_train=s.height, n_rank=rank_n, share_gt1h=round((s["TAXITIME_SEC_mvt"] > 3600).mean(), 4), recall_gt1h=round((s["TAXITIME_SEC_mvt"] > 3600).sum() / tot1, 3),
                     recall_gt3h=round((s["TAXITIME_SEC_mvt"] > 3 * 3600).sum() / tot3, 3), sched_share=round((s["clock_src"] == "sched").mean(), 3),
                     rmse_if_pred_sched=round(float(np.sqrt(np.mean((s["m_s"].to_numpy() - s["TAXITIME_SEC_mvt"].to_numpy()) ** 2))), 0),
                     rmse_if_pred_median=round(float(np.sqrt(np.mean((np.median(s["TAXITIME_SEC_mvt"].to_numpy()) - s["TAXITIME_SEC_mvt"].to_numpy()) ** 2))), 0)))
print(pl.DataFrame(rows))
print("y>3h rows:", tot3, " y>1h:", tot1)
