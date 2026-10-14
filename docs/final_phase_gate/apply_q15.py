"""LIRF copy-overlay re-weight (0.4*q -> q**1.5) applied as a delta table on top of a final candidate file.

    python -I docs/final_phase_gate/apply_q15.py <delta_table.parquet> <in.parquet> <out.parquet>

<delta_table>: MVT_ID_mvt, old, new, delta computed on the integrator's v4 (final_recommended, md5 d0ce32b6) by the
sweep script (p_pre = (v4 - 0.4*q*ms)/(1 - 0.4*q); new = p_pre + q**1.5*(ms - p_pre), reverted to p_pre if negative,
LOBT clip, floor 0). The delta is ADDED to <in> (which may already carry the LIRF gate and/or the Jan/Jul +1*delta
dither: the gate rows are disjoint from these rows, and the dither shifts both old and new by the same delta), then
the LOBT identity clip [ceil(mml-3606), ceil(mml+3606)] on the final clock and the zero floor are re-applied once.
Never invert the overlay from a dithered file."""
import sys
import numpy as np
import polars as pl

K, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
tab, inp, outp = sys.argv[1:4]
D = pl.read_parquet(tab).select(K, pl.col("delta").cast(pl.Float64))
F = pl.read_parquet(inp)
raw = (pl.scan_parquet("data/raw/final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
       .select(K, "ADEP_mvt", pl.col("AOBT_3_flt").is_not_null().alias("matched"),
               (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mml"))
       .collect())
Z = F.join(D, on=K, how="left", maintain_order="left").join(raw, on=K, how="left", maintain_order="left")
assert Z.height == F.height and Z["ADEP_mvt"].null_count() == 0
touched = Z.filter(pl.col("delta").is_not_null())
assert (touched["ADEP_mvt"] == "LIRF").all() and touched["matched"].all(), "delta rows must be matched LIRF rows"
p = pl.col(T).cast(pl.Float64)
raw_new = p + pl.col("delta").fill_null(0.0)
lo, hi = (pl.col("mml") - 3606).ceil(), (pl.col("mml") + 3606).ceil()
clipped = pl.when(pl.col("mml").is_null()).then(raw_new).otherwise(pl.max_horizontal(pl.min_horizontal(raw_new, hi), lo))
out = Z.select(K, clipped.clip(lower_bound=0).round(0).cast(pl.Int32).alias(T))
assert out.height == F.height and out[T].null_count() == 0 and bool((out[T] >= 0).all())
assert out[K].equals(F[K])
out.write_parquet(outp)
d = (out[T].cast(pl.Float64) - F[T].cast(pl.Float64)).to_numpy()
viol = Z.with_columns(n=out[T].cast(pl.Float64)).filter(pl.col("mml").is_not_null() & ((pl.col("n") < lo) | (pl.col("n") > hi))).height
print(f"rows changed {int((d != 0).sum()):,}; sum d^2/N {float((d ** 2).sum()) / len(d):.1f} s^2; max|d| {int(np.abs(d).max())}; "
      f"delta rows {D.height:,}; LOBT-window violations after {viol}; wrote {outp}")
