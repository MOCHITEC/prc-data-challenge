"""Disruption days: on days where the observable MVT-AOBT3 level is far above the airport's norm, how does the
UNOBSERVABLE part (y - (MVT-AOBT3) = AOBT3 - BLOCK) behave? Is the relation shared across airports (so a pooled,
airport-normalised feature could transfer to EHAM 2026-01-02..07 which has no 2025 analogue at EHAM)?

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

f = features().with_columns(pl.col("APT").cast(pl.Utf8))
ok = pl.col("y").is_between(0, 7200) & pl.col("m_aobt3").is_between(0, 7200)
day = f.group_by("set", "APT", "date_local").agg(
    pl.len().alias("n"),
    pl.col("day_m_aobt3_mean").first(), pl.col("day_msched_gt3h").first(), pl.col("day_nm_unmatched").first(),
    pl.col("day_dep_n_rel").first(), pl.col("day_arr_taxi_in_mean").first(),
    (pl.col("y") - pl.col("m_aobt3")).filter(ok).mean().alias("gap"),   # = AOBT3 - BLOCK, unobservable
    pl.col("y").filter(pl.col("y").is_between(0, 7200)).mean().alias("y_mean"),
)
norm = day.filter(pl.col("set") == "train").group_by("APT").agg(
    pl.col("day_m_aobt3_mean").median().alias("ma_norm"), pl.col("gap").median().alias("gap_norm"),
    pl.col("day_arr_taxi_in_mean").median().alias("ti_norm"))
day = day.join(norm, on="APT").with_columns(
    (pl.col("day_m_aobt3_mean") / pl.col("ma_norm")).alias("ma_ratio"),
    (pl.col("day_arr_taxi_in_mean") / pl.col("ti_norm")).alias("ti_ratio"),
    (pl.col("gap") - pl.col("gap_norm")).alias("gap_dev"))
tr = day.filter(pl.col("set") == "train")

print("\n## A. 2025 airport-days binned by ma_ratio = day mean(MVT-AOBT3) / airport norm: excess unobservable gap")
bins = [0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 2.0]
t = tr.with_columns(pl.col("ma_ratio").cut(bins).alias("bin")).group_by("bin").agg(
    pl.len().alias("days"), pl.col("gap_dev").mean().alias("gap_dev_mean"), pl.col("gap_dev").median().alias("gap_dev_med"),
    pl.col("gap_dev").std().alias("gap_dev_sd"), pl.col("APT").unique().sort().str.join(",").alias("airports")).sort("bin")
print(t)

print("\n## B. same, split by airport for ma_ratio > 1.2 (disruption days)")
print(tr.filter(pl.col("ma_ratio") > 1.2).group_by("APT").agg(pl.len().alias("days"), pl.col("gap_dev").mean(),
                                                               pl.col("ma_ratio").mean()).sort("APT"))

print("\n## C. pooled OLS gap_dev ~ (ma_ratio-1) + msched_gt3h + nm_unmatched + (ti_ratio-1), leave-one-airport-out")
X_cols = ["ma_ratio", "day_msched_gt3h", "day_nm_unmatched", "ti_ratio"]
g = tr.drop_nulls(X_cols + ["gap_dev"]).filter(pl.col("n") > 50)
X = np.column_stack([np.ones(g.height)] + [g[c].to_numpy() - (1.0 if "ratio" in c else 0.0) for c in X_cols])
yv = g["gap_dev"].to_numpy()
apt = g["APT"].to_numpy()
b_all = np.linalg.lstsq(X, yv, rcond=None)[0]
print("pooled coef (const, ma_ratio-1, msched_gt3h, nm_unmatched, ti_ratio-1):", np.round(b_all, 1))
rows = []
for a in sorted(set(apt)):
    b = np.linalg.lstsq(X[apt != a], yv[apt != a], rcond=None)[0]
    p = X[apt == a] @ b
    yy = yv[apt == a]
    big = g.filter(pl.col("APT") == a)["ma_ratio"].to_numpy() > 1.2
    rows.append({"APT": a, "days": int((apt == a).sum()), "R2_loao": 1 - np.sum((yy - p) ** 2) / np.sum((yy - yy.mean()) ** 2),
                 "disr_days": int(big.sum()), "gap_dev_disr": float(yy[big].mean()) if big.any() else None,
                 "pred_disr": float(p[big].mean()) if big.any() else None})
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(2)))

print("\n## D. ranking days with ma_ratio > 1.2: rows and pooled-model predicted excess gap")
rk = day.filter(pl.col("set") == "rank").drop_nulls(X_cols)
Xr = np.column_stack([np.ones(rk.height)] + [rk[c].to_numpy() - (1.0 if "ratio" in c else 0.0) for c in X_cols])
rk = rk.with_columns(pl.Series("pred_gap_dev", Xr @ b_all))
sel = rk.filter(pl.col("ma_ratio") > 1.2).sort("APT", "date_local")
print(sel.select("APT", "date_local", "n", "ma_ratio", "day_msched_gt3h", "day_nm_unmatched", "ti_ratio", "pred_gap_dev"))
print("ranking DEP rows on such days:", sel["n"].sum(), "of", rk["n"].sum(),
      "; 2025 train rows on such days:", tr.filter(pl.col("ma_ratio") > 1.2)["n"].sum())
# what would an uncorrected model lose there? SSE if the true excess gap equals the pooled prediction
sse = float((sel["n"] * sel["pred_gap_dev"] ** 2).sum())
tot_n = rk["n"].sum()
print(f"if missed entirely: added MSE {sse / tot_n:.0f} s^2 -> RMSE 305 -> {np.sqrt(305**2 + sse / tot_n):.1f}")

print("\n## E. robust version: bins of day share of departures with MVT-SCHED > 3h (a disruption marker that is not"
      " confounded by runway configuration the way MVT-AOBT3 is at EHAM)")
b2 = [0.01, 0.02, 0.05, 0.1, 0.2]
for name, frame in (("train 2025", tr), ("rank 2026", day.filter(pl.col("set") == "rank"))):
    t = frame.with_columns(pl.col("day_msched_gt3h").cut(b2).alias("bin")).group_by("bin").agg(
        pl.len().alias("days"), pl.col("n").sum().alias("rows"), pl.col("gap_dev").mean().alias("gap_dev_mean"),
        pl.col("gap_dev").median().alias("gap_dev_med"), pl.col("ma_ratio").mean().alias("ma_ratio"),
        pl.col("APT").unique().sort().str.join(",").alias("airports")).sort("bin")
    print(f"### {name}")
    print(t)
