"""Day-level anomalies: which airport-days carry the error, and are they visible from ranking-observable day
aggregates? Also lists the ranking (2026) days that look anomalous by the same observables.

Output: markdown tables (docs/analysis/temporal_shift.md §4).
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from temporal_shift_features import features  # noqa: E402

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_hide_dataframe_shape(True)

OBS = ["day_m_aobt3_mean", "day_arr_taxi_in_mean", "day_dep_n_rel", "day_nm_unmatched", "day_msched_gt3h",
       "day_arr_blk_eq_sched", "day_aobt3_eq_sched"]
f = features()
f = f.with_columns(pl.col("APT").cast(pl.Utf8))
day = f.group_by("set", "APT", "date_local").agg(
    pl.len().alias("n"),
    pl.col("y").filter(pl.col("y").is_between(0, 7200)).mean().alias("y_mean_clean"),
    pl.col("y").mean().alias("y_mean"),
    (pl.col("y") ** 2).sum().alias("y_ss"),
    (pl.col("y") > 3 * 3600).sum().alias("n_gt3h"),
    (pl.col("clock_src") == "sched").mean().alias("src_sched"),
    (pl.col("m_aobt3") - pl.col("y")).filter(pl.col("y").is_between(0, 7200)).mean().alias("offset"),
    *[pl.col(c).first() for c in OBS],
)
tr = day.filter(pl.col("set") == "train").with_columns(
    ((pl.col("y_mean_clean") - pl.col("y_mean_clean").mean().over("APT")) / pl.col("y_mean_clean").std().over("APT"))
    .alias("z_y"),
    ((pl.col("offset") - pl.col("offset").mean().over("APT")) / pl.col("offset").std().over("APT")).alias("z_offset"))

print("\n## A. top 25 anomalous airport-days in 2025 by |z| of daily clean y mean")
print(tr.sort(pl.col("z_y").abs(), descending=True).head(25).select(
    "APT", "date_local", "n", "y_mean_clean", "z_y", "offset", "z_offset", "n_gt3h", "src_sched", *OBS))

print("\n## B. top 20 airport-days by count of y > 3h (the RMSE tail)")
print(tr.sort("n_gt3h", descending=True).head(20).select("APT", "date_local", "n", "n_gt3h", "y_mean", "src_sched",
                                                          "day_nm_unmatched", "day_msched_gt3h", "day_dep_n_rel"))
tot = tr["n_gt3h"].sum()
top = tr.sort("n_gt3h", descending=True)
print(f"y>3h rows total {tot}; share in top 10 days {top.head(10)['n_gt3h'].sum() / tot:.3f}; "
      f"days with >=1: {(tr['n_gt3h'] > 0).sum()}")

print("\n## C. Can the day level be predicted from observables? within-airport corr of z_y with each observable")
rows = []
for apt in sorted(tr["APT"].unique().to_list()):
    g = tr.filter(pl.col("APT") == apt).drop_nulls(OBS + ["z_y"])
    r = {"APT": apt}
    for c in OBS:
        x = g[c].to_numpy()
        r[c.replace("day_", "")] = float(np.corrcoef(x, g["z_y"].to_numpy())[0, 1]) if x.std() > 0 else None
    # OLS R2 in-sample and leave-one-month-out
    X = np.column_stack([np.ones(g.height)] + [g[c].to_numpy() for c in OBS])
    yv = g["z_y"].to_numpy()
    mo = g["date_local"].dt.month().to_numpy()
    pred = np.empty_like(yv)
    for mm in np.unique(mo):
        b = np.linalg.lstsq(X[mo != mm], yv[mo != mm], rcond=None)[0]
        pred[mo == mm] = X[mo == mm] @ b
    r["R2_lomo"] = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
    # same for the unobservable offset (BLOCK - AOBT3) : is it observable through the day aggregates?
    ov = g["z_offset"].to_numpy()
    po = np.empty_like(ov)
    for mm in np.unique(mo):
        b = np.linalg.lstsq(X[mo != mm], ov[mo != mm], rcond=None)[0]
        po[mo == mm] = X[mo == mm] @ b
    r["R2_lomo_offset"] = 1 - np.sum((ov - po) ** 2) / np.sum((ov - ov.mean()) ** 2)
    rows.append(r)
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(2)))

print("\n## D. ranking (2026) days that are extreme on the observables, relative to the 2025 distribution")
ref = tr.group_by("APT").agg([pl.col(c).quantile(0.99).alias(f"{c}_p99") for c in OBS])
rk = day.filter(pl.col("set") == "rank").join(ref, on="APT")
rk = rk.with_columns(pl.sum_horizontal([(pl.col(c) > pl.col(f"{c}_p99")).cast(pl.Int8) for c in
                                        ("day_m_aobt3_mean", "day_nm_unmatched", "day_msched_gt3h",
                                         "day_arr_taxi_in_mean")]).alias("n_extreme"))
print(rk.filter(pl.col("n_extreme") >= 2).sort("APT", "date_local").select("APT", "date_local", "n", "n_extreme", *OBS))
tr2 = tr.join(ref, on="APT").with_columns(pl.sum_horizontal(
    [(pl.col(c) > pl.col(f"{c}_p99")).cast(pl.Int8) for c in
     ("day_m_aobt3_mean", "day_nm_unmatched", "day_msched_gt3h", "day_arr_taxi_in_mean")]).alias("n_extreme"))
print("2025 days with n_extreme>=2:", tr2.filter(pl.col("n_extreme") >= 2).height, "; 2026 days:",
      rk.filter(pl.col("n_extreme") >= 2).height, "(of", rk.height, ")")
print("\n## E. 2025 days with n_extreme>=2: their y level")
print(tr2.filter(pl.col("n_extreme") >= 2).sort("APT", "date_local").select(
    "APT", "date_local", "n", "y_mean_clean", "z_y", "offset", "n_gt3h", *OBS))
