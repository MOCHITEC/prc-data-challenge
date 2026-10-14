"""Both LIRF corrections on any base that carries the integrator's 0.4*q copy overlay (unfixed q) and LOBT clip:

    python -I docs/final_phase_gate/build_on_base.py <base.parquet> <base_plus1delta.parquet> <out_dir> <tag>

1) gate: the copy overlay removed on LIRF matched rows with |LOBT - SCHED| > 3606 s. Delta per row = new - old of the
   gate table (computed exactly on v2, handoff/2026-10-14/lirf_copy_gate_rows_v2.parquet), added to the base.
2) q15: on the other LIRF matched rows with a q (copyq_LIRF_final_unfixed), the pull 0.4*q is replaced by q**1.5:
   p_pre = (base - 0.4*q*ms)/(1 - 0.4*q); new = p_pre + q**1.5*(ms - p_pre), reverted to p_pre if negative. Inverted
   from the UNDITHERED base only.
Both deltas are then added to the base and to the integrator's +1x-dither file of the same base, followed by the LOBT
identity clip [ceil(mml-3606), ceil(mml+3606)] on the final clock and the zero floor. The dither step is re-implemented
and checked against the given +1x file (must be identical)."""
import os
import sys
import numpy as np
import polars as pl

K, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
base_p, dith_p, out_dir, tag = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "..", "..", "handoff", "2026-10-14", "lirf_copy_gate_rows_v2.parquet")
QFILE = os.environ.get("QFILE", "/home/user/lc-final/handoff/2026-10-13/inputs/copyq_LIRF_final_unfixed.parquet")
B = pl.read_parquet(base_p)
DI = pl.read_parquet(dith_p)
R = (pl.scan_parquet("data/raw/final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
     .select(K, "ADEP_mvt", pl.col("AOBT_3_flt").is_not_null().alias("mat"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon"),
             (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mml"),
             (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("ms"))
     .collect())
lo, hi = (pl.col("mml") - 3606).ceil(), (pl.col("mml") + 3606).ceil()

# gate delta
G = pl.read_parquet(GATE)
X = G.join(R, on=K)
assert X.height == G.height == 2486 and (X["ADEP_mvt"] == "LIRF").all() and X["mat"].all()
assert ((X["ms"] - X["mml"]).abs() > 3606).all()
GD = G.select(K, (pl.col("new") - pl.col("old")).cast(pl.Float64).alias("gd"))

# q15 delta, inverted from the undithered base
Q = pl.read_parquet(QFILE).select(K, "q")
L = (B.join(R, on=K).join(Q, on=K, how="inner")
     .filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("mat") & ((pl.col("ms") - pl.col("mml")).abs() <= 3606)))
p, q, ms = pl.col(T).cast(pl.Float64), pl.col("q"), pl.col("ms")
ppre = (p - 0.4 * q * ms) / (1 - 0.4 * q)
pn = ppre + q ** 1.5 * (ms - ppre)
pn = pl.when(pn < 0).then(ppre).otherwise(pn)
pn = pl.max_horizontal(pl.min_horizontal(pn, hi), lo).clip(lower_bound=0).round(0)
QD = L.with_columns(new=pn).select(K, (pl.col("new") - p).alias("qd")).filter(pl.col("qd") != 0)
assert QD.join(GD, on=K, how="inner").height == 0, "q15 rows overlap the gate rows"


def add(sub, *deltas):
    Z = sub.join(R.select(K, "mml"), on=K, how="left", maintain_order="left")
    tot = pl.lit(0.0)
    for name, D in deltas:
        Z = Z.join(D, on=K, how="left", maintain_order="left")
        tot = tot + pl.col(name).fill_null(0.0)
    touched = pl.any_horizontal([pl.col(n).is_not_null() for n, _ in deltas])
    v = pl.col(T).cast(pl.Float64) + tot
    v = pl.when(touched & pl.col("mml").is_not_null()).then(pl.max_horizontal(pl.min_horizontal(v, hi), lo)).otherwise(v)
    out = Z.select(K, pl.when(touched).then(v.clip(lower_bound=0).round(0)).otherwise(pl.col(T).cast(pl.Float64)).cast(pl.Int32).alias(T))
    assert out[K].equals(sub[K]) and out[T].null_count() == 0 and bool((out[T] >= 0).all())
    return out


def dither(sub):
    old = (pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
           .select(K, pl.col("MVT_TIME_UTC_mvt").alias("t_old")).collect())
    j = (R.select(K, "mat").join(pl.scan_parquet("data/raw/final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
                                  .select(K, "MVT_TIME_UTC_mvt").collect(), on=K)
         .join(old, on=K).filter("mat").select(K, (pl.col("MVT_TIME_UTC_mvt") - pl.col("t_old")).dt.total_seconds().cast(pl.Float64).alias("dl")))
    return add(sub, ("dl", j))


mine = dither(B)
print("dither re-implementation == given +1x file:", mine.equals(DI))
assert mine.equals(DI), "the +1x file is not the base dithered once"
tpl = pl.read_parquet("data/raw/final_submitting.parquet")
outs = {
    f"final_{tag}_gatefull": add(B, ("gd", GD)),
    f"final_{tag}_gatefull_plus1delta": add(DI, ("gd", GD)),
    f"final_{tag}_gatefull_q15": add(B, ("gd", GD), ("qd", QD)),
    f"final_{tag}_gatefull_q15_plus1delta": add(DI, ("gd", GD), ("qd", QD)),
    f"final_{tag}_q15_plus1delta": add(DI, ("qd", QD)),
}
mon = B.join(R.select(K, "mon"), on=K, how="left", maintain_order="left")["mon"].to_numpy()
for name, f in outs.items():
    f = tpl.select(K).join(f, on=K, how="left", maintain_order="left").cast(tpl.schema)
    assert f.height == tpl.height and f.null_count().sum_horizontal().item() == 0
    f.write_parquet(os.path.join(out_dir, f"{name}.parquet"))
    ref = DI if name.endswith("plus1delta") else B
    d = (f[T].cast(pl.Float64) - ref[T].cast(pl.Float64)).to_numpy()
    viol = (f.join(R.select(K, "mml"), on=K).filter(pl.col("mml").is_not_null() & ((pl.col(T) < lo) | (pl.col(T) > hi))).height)
    print(f"{name}: vs {'dithered' if ref is DI else 'plain'} base: rows changed {int((d != 0).sum()):,} "
          f"({ {m: int(((d != 0) & (mon == m)).sum()) for m in (1, 2, 6, 7)} }), sum d^2/N {float((d ** 2).sum()) / len(d):.1f}, "
          f"max|d| {int(np.abs(d).max())}, LOBT violations {viol}")
QD.write_parquet(os.path.join(out_dir, f"lirf_overlay_q15_rows_{tag}.parquet"))
print("q15 delta rows", QD.height, "gate rows", GD.height)
