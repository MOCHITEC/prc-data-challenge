"""adsbaudit step 1: base b, ADS-B proxy and residual c, 2025 training rows vs 2026 rows (local, read-only).

2025: $SCRATCH/learn/cv_rows_months12_v2.parquet (12-month rows; b = the OOF base the residual was trained on; c_C =
12:v2 leave-days-out correction) joined to the four 2025 feature files. 2026: $SCRATCH/agents/rescat/rank_delta_all.parquet
(b recovered from v40 by the production fixed point; c_prod = file - b is the residual v40 carries) joined to feat_2026.
Ground-fix rows only, LTFM and LIRF dropped (production never applies the current residual there).
proxy p = t_mvt - t_move (ADS-B taxi estimate). Writes $SCRATCH/agents/adsbaudit/bc.parquet and prints tables.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsbaudit_bc.py
"""
import os

import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
OUT = os.path.join(S, "agents", "adsbaudit")
os.makedirs(OUT, exist_ok=True)
L = os.path.join(S, "learn")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
FC = ["MVT_ID_mvt", "t_mvt", "t_aobt3", "t_move", "t_move_lo", "t_push", "ground_fix", "day"]
F25 = pl.concat([pl.read_parquet(os.path.join(L, f), columns=FC) for f in
                 ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet", "feat_2025fmj.parquet")])
R25 = pl.read_parquet(os.path.join(L, "cv_rows_months12_v2.parquet")).filter("has").join(F25, on="MVT_ID_mvt")
R25 = R25.filter(pl.col("ground_fix")).select(
    "MVT_ID_mvt", "ADEP_mvt", "y", "b", pl.col("c_C").alias("c"), "t_mvt", "t_aobt3", "t_move", "t_move_lo", "t_push",
    pl.when(pl.col("ym").is_in(["2025-01", "2025-07"])).then(pl.col("ym")).otherwise(pl.lit("2025-oth")).alias("per"))
F26 = pl.read_parquet(os.path.join(L, "feat_2026.parquet"), columns=FC)
R26 = pl.read_parquet(os.path.join(S, "agents", "rescat", "rank_delta_all.parquet")).join(F26, on="MVT_ID_mvt")
R26 = R26.select("MVT_ID_mvt", "ADEP_mvt", pl.lit(None, pl.Float64).alias("y"), "b", (pl.col("file") - pl.col("b")).alias("c"),
                 "t_mvt", "t_aobt3", "t_move", "t_move_lo", "t_push", pl.col("day").str.slice(0, 7).alias("per"))
D = pl.concat([R25, R26]).filter(~pl.col("ADEP_mvt").is_in(["LTFM", "LIRF"])).with_columns(
    pl.when(pl.col("t_aobt3").is_not_null()).then(pl.lit("A")).otherwise(pl.lit("B")).alias("lane"),
    (pl.col("t_mvt") - pl.col("t_move")).alias("p"))
D = D.with_columns((pl.col("b") - pl.col("p")).alias("bp"), (pl.col("y") - pl.col("b")).alias("yb"))
D.write_parquet(os.path.join(OUT, "bc.parquet"))


def summ(G):
    return G.agg(pl.len().alias("n"), pl.col("b").mean().round(0).alias("b_mean"), pl.col("p").median().round(0).alias("p_med"),
                 pl.col("bp").median().round(0).alias("bp_med"),
                 (pl.col("bp").quantile(0.75) - pl.col("bp").quantile(0.25)).round(0).alias("bp_iqr"),
                 pl.col("bp").filter(pl.col("bp").abs() < 1800).std().round(0).alias("bp_sd"),
                 pl.col("c").mean().round(1).alias("c_mean"), pl.col("c").std().round(1).alias("c_sd"),
                 (pl.col("c").abs() > 120).mean().round(3).alias("c>120"), (pl.col("c").abs() > 300).mean().round(4).alias("c>300"),
                 pl.col("yb").mean().round(1).alias("yb_mean"))


print("per period x lane (ground-fix rows, LTFM/LIRF dropped)")
print(summ(D.group_by("per", "lane")).sort("lane", "per"))
print("\nper airport x lane A, by period")
T = summ(D.filter(pl.col("lane") == "A").group_by("ADEP_mvt", "per")).sort("ADEP_mvt", "per")
print(T.select("ADEP_mvt", "per", "n", "b_mean", "p_med", "bp_med", "bp_iqr", "bp_sd", "c_mean", "c_sd", "c>120", "yb_mean"))
print("\nper airport x lane B, by period")
T = summ(D.filter(pl.col("lane") == "B").group_by("ADEP_mvt", "per")).sort("ADEP_mvt", "per")
print(T.select("ADEP_mvt", "per", "n", "b_mean", "p_med", "bp_med", "bp_iqr", "bp_sd", "c_mean", "c_sd", "c>120", "yb_mean"))
# c as a function of b - p: slope of c on (p - b) within |p - b| < 900
print("\nslope of c on (p - b) [|p-b|<900] and mean c by (p-b) bin, lane A")
for per in ["2025-01", "2025-07", "2025-oth", "2026-01", "2026-07"]:
    X = D.filter((pl.col("per") == per) & (pl.col("lane") == "A") & (pl.col("bp").abs() < 900)).drop_nulls("bp")
    x, c = (-X["bp"]).to_numpy(), X["c"].to_numpy()
    sl = np.polyfit(x, c, 1)[0]
    bins = X.with_columns((-pl.col("bp")).cut([-300, -120, -60, 0, 60, 120, 300]).alias("bin")).group_by("bin").agg(
        pl.len(), pl.col("c").mean().round(0), pl.col("yb").mean().round(0)).sort("bin")
    print(per, f"slope {sl:.3f}", bins.rows())
