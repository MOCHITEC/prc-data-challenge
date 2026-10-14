"""Old-board (Jan/Jul 2026, old clock) test files for the two LIRF corrections, so the submitter can check them on
the ranking leaderboard before the final upload:

    python -I docs/final_phase_gate/oldboard_test.py <v48.parquet> <q_oldframe.parquet> <out_dir>

v50 = v48 + gate (copy overlay removed on LIRF matched rows with |LOBT - SCHED| > 3606 s)
v51 = v50 + q^1.5 re-weight of the copy overlay on the other LIRF matched rows
Inversion: p_pre = (v48 - 0.4*q*ms)/(1 - 0.4*q) with q = q_orig (the unfixed Jan+Jul frame used by v47/v48), ms = MVT - SCHED
of the old ranking file; then LOBT clip on the old clock and the zero floor. The inversion is approximate (the v48 chain
reproduces the overlay to RMS about 4 s)."""
import os
import sys
import numpy as np
import polars as pl

K, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
v48p, qp, out = sys.argv[1:4]
V = pl.read_parquet(v48p)
Q = pl.read_parquet(qp).select(K, pl.col("ms").cast(pl.Float64).alias("ms_q"), pl.col("q_orig").alias("q"))
R = (pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
     .select(K, "ADEP_mvt", pl.col("AOBT_3_flt").is_not_null().alias("matched"),
             (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ms"),
             (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mml"),
             pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect())
Z = V.join(R, on=K, how="left", maintain_order="left").join(Q, on=K, how="left", maintain_order="left")
assert Z.height == V.height == 344841 and Z["ADEP_mvt"].null_count() == 0
L = Z.filter(pl.col("q").is_not_null())
assert (L["ADEP_mvt"] == "LIRF").all() and L["matched"].all() and L["mml"].null_count() == 0
print("q rows", L.height, "ms(q file) == ms(ranking):", bool((L["ms_q"] == L["ms"]).all()), "max |diff|", float((L["ms_q"] - L["ms"]).abs().max()))
n_lirf = Z.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("matched")).height
print("LIRF matched rows in ranking:", n_lirf, "(q rows cover", L.height, ")")
p, q, ms = pl.col(T).cast(pl.Float64), pl.col("q"), pl.col("ms")
imp = (ms - pl.col("mml")).abs() > 3606
ppre = (p - 0.4 * q * ms) / (1 - 0.4 * q)
lo, hi = (pl.col("mml") - 3606).ceil(), (pl.col("mml") + 3606).ceil()
def finish(expr):
    e = pl.when(expr < 0).then(ppre).otherwise(expr)
    e = pl.max_horizontal(pl.min_horizontal(e, hi), lo).clip(lower_bound=0).round(0)
    return pl.when(q.is_not_null()).then(e).otherwise(p).cast(pl.Int32)
# v50: gate only (impossible rows -> p_pre, others keep the 0.4 q overlay exactly as in v48)
v50 = finish(pl.when(imp).then(ppre).otherwise(p))
# v51: gate + q^1.5 on the possible rows
v51 = finish(pl.when(imp).then(ppre).otherwise(ppre + q ** 1.5 * (ms - ppre)))
O = Z.with_columns(v50=v50, v51=v51)
for name in ("v50", "v51"):
    f = O.select(K, pl.col(name).alias(T))
    assert f[T].null_count() == 0 and bool((f[T] >= 0).all()) and f[K].equals(V[K])
    f.write_parquet(os.path.join(out, f"{name}_oldboard.parquet"))
    d = (f[T].cast(pl.Float64) - V[T].cast(pl.Float64)).to_numpy()
    bym = O.with_columns(d=(pl.col(name) - pl.col(T)).cast(pl.Float64)).filter(pl.col("d") != 0).group_by("mon").agg(
        n=pl.len(), sumd2=(pl.col("d") ** 2).sum() / 1e6, mean=pl.col("d").mean().round(1)).sort("mon")
    viol = O.filter(pl.col("mml").is_not_null() & ((pl.col(name) < lo) | (pl.col(name) > hi))).height
    print(f"{name}: rows changed vs v48 {int((d != 0).sum()):,}; sum d^2 / 344,841 = {float((d ** 2).sum()) / 344841:.1f} s^2; max|d| {int(np.abs(d).max())}; LOBT violations {viol}")
    print(bym)
d = (O["v51"] - O["v50"]).cast(pl.Float64).to_numpy()
print(f"v51 - v50: rows {int((d != 0).sum()):,}; sum d^2 / 344,841 = {float((d ** 2).sum()) / 344841:.1f} s^2; max|d| {int(np.abs(d).max())}")
