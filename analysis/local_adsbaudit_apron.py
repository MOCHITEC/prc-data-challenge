"""adsbaudit: did the 2026 base already absorb the per-apron bias that the apron residual corrects? (local, read-only)

Per apron (ground-fix rows, matched lane, not LTFM/LIRF, >= 300 rows each year): 2025 (7 months, rescat cv_rows) the
apron bias the residual can learn, bias25 = mean(y - b - c_A0), the 2025 CV apron delta d25 = mean(c_A3 - c_A0), and
the proxy gap m25 = mean(p - b - c_A0) with p = t_mvt - t_move; 2026 (v40 file f = b + c_prod, apron delta of the
v43 swap) d26 and m26 = mean(p - f). If p - y per apron is stable across years, bias26 ~ bias25 + (m26 - m25).

    POLARS_MAX_THREADS=1 .venv/bin/python analysis/local_adsbaudit_apron.py
"""
import os

import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
LD = os.path.join(S, "learn")
RS = os.path.join(S, "agents", "rescat")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
codes = pl.read_parquet(os.path.join(RS, "codes.parquet")).select("MVT_ID_mvt", "apron")
FC = ["MVT_ID_mvt", "t_mvt", "t_move", "t_aobt3", "ground_fix"]
F25 = pl.concat([pl.read_parquet(os.path.join(LD, f), columns=FC) for f in ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet")])
A = (pl.read_parquet(os.path.join(RS, "cv_rows.parquet")).join(F25.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)), on="MVT_ID_mvt")
     .join(codes, on="MVT_ID_mvt"))
A = A.filter(pl.col("ground_fix") & pl.col("t_aobt3").is_not_null() & ~pl.col("ADEP_mvt").is_in(["LTFM", "LIRF"]))
T25 = A.group_by("apron").agg(pl.len().alias("n25"), (pl.col("y") - pl.col("b") - pl.col("c_A0")).mean().alias("bias25"),
                              (pl.col("c_A3") - pl.col("c_A0")).mean().alias("d25"),
                              (pl.col("t_mvt") - pl.col("t_move") - pl.col("b") - pl.col("c_A0")).mean().alias("m25"))
F26 = pl.read_parquet(os.path.join(LD, "feat_2026.parquet"), columns=FC).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
dl = pl.read_parquet("/home/user/prc-data-challenge/outputs/parts/apron_delta_12m.parquet").select("MVT_ID_mvt", "delta")
B = (pl.read_parquet(os.path.join(RS, "rank_delta_all.parquet")).select("MVT_ID_mvt", "ADEP_mvt", "apron", "file")
     .join(F26, on="MVT_ID_mvt").join(dl, on="MVT_ID_mvt", how="left"))
B = B.filter(pl.col("ground_fix") & pl.col("t_aobt3").is_not_null() & ~pl.col("ADEP_mvt").is_in(["LTFM", "LIRF"]))
T26 = B.group_by("apron").agg(pl.len().alias("n26"), pl.col("delta").fill_null(0).mean().alias("d26"),
                              (pl.col("t_mvt") - pl.col("t_move") - pl.col("file")).mean().alias("m26"))
T = T25.join(T26, on="apron").filter((pl.col("n25") >= 300) & (pl.col("n26") >= 300)).with_columns(
    (pl.col("m26") - pl.col("m25")).alias("dm"), (pl.col("bias25") + pl.col("m26") - pl.col("m25")).alias("bias26_est"))
w = np.sqrt(T["n26"].to_numpy())
def wc(a, b):
    a, b = T[a].to_numpy(), T[b].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)
print("aprons", T.height, "| corr(d25, bias25)", wc("d25", "bias25"), "| corr(d26, d25)", wc("d26", "d25"),
      "| corr(d26, bias26_est)", wc("d26", "bias26_est"), "| corr(d25, dm)", wc("d25", "dm"))
n = T["n26"].to_numpy()
for col in ("bias25", "d25", "d26", "bias26_est", "dm"):
    v = T[col].to_numpy()
    print(f"  row-weighted RMS over aprons of {col:10s}: {np.sqrt(np.nansum(n * v ** 2) / n[np.isfinite(v)].sum()):.1f}")
# slope of bias26_est on d26: 1 = the 2026 bias is still there, 0 = gone
x, yv = T["d26"].to_numpy(), T["bias26_est"].to_numpy()
ok = np.isfinite(x) & np.isfinite(yv)
print("weighted slope bias25 on d25:", round(float(np.polyfit(T["d25"].to_numpy(), T["bias25"].to_numpy(), 1, w=w)[0]), 2),
      "| bias26_est on d26:", round(float(np.polyfit(x[ok], yv[ok], 1, w=w[ok])[0]), 2))
print(T.sort("n26", descending=True).head(25).with_columns(pl.selectors.float().round(1)))
