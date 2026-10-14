"""Re-weight the LIRF unmatched copy mixture (ucmix) on a final file: w = 0.5 -> W.

    python -I docs/final_phase_gate/apply_ucmix_w.py <ucmix_delta.parquet> <W> <in.parquet> <out.parquet>

ucmix adds d = w*(M - plain) on LIRF unmatched rows with 0 < MVT - SCHED < 8 h (M = p*ms + (1-p)*g_nc), skipping the
rows that the LIRF unmatched rules B/C claim (same rule definitions as laughing-cray apply_overlays.unmatched_rules,
on final_ranking with the 2025 prefix rates). With w = 0.5 in production, the file at weight W is in + (W/0.5 - 1)*d.
Unmatched rows have no LOBT window; the zero floor is re-applied."""
import glob
import sys
import numpy as np
import polars as pl

K, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
dpath, W, inp, outp = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "TAXITIME_SEC_mvt"]
off = 3600 + pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.dst_offset().dt.total_seconds()


def lirf_unmatched(df):
    return (df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
            .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                          pre=pl.col("FLIGHT_mvt").str.slice(0, 3).fill_null("?"), off=off))


tr = lirf_unmatched(pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/raw/training_*.parquet"))]))
small = tr.filter(pl.col("ms").is_between(0, 3599)).with_columns(
    tz=((pl.col("TAXITIME_SEC_mvt") - pl.col("off")).is_between(240, 2700)) & ((pl.col("TAXITIME_SEC_mvt") - pl.col("ms")).abs() > 12))
qp = small.group_by("pre").agg(t=pl.col("tz").sum(), n=pl.len()).filter(pl.col("t") >= 1)
R = lirf_unmatched(pl.read_parquet("data/raw/final_ranking.parquet", columns=cols[:-1])).join(qp, on="pre", how="left")
R = R.with_columns(dh=(pl.col("ms") - 1100) / 3600, qtz=(pl.col("t") / (pl.col("n") + 1)).fill_null(0.0))
rule = (pl.when(pl.col("ms") >= 50400).then(pl.lit("-")).when(pl.col("dh").is_between(7, 13.5)).then(pl.lit("C"))
        .when(pl.col("ms").is_between(0, 3599) & (pl.col("qtz") > 0)).then(pl.lit("B")).otherwise(pl.lit("-")))
R = R.with_columns(rule=rule)
claimed = R.filter(pl.col("rule") != "-")[K]
D = (pl.read_parquet(dpath).filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("delta") != 0))
     .filter(~pl.col(K).is_in(claimed.implode())).select(K, pl.col("delta").alias("d")))
assert D.join(R.select(K, "ms"), on=K).filter((pl.col("ms") <= 0) | (pl.col("ms") >= 28800)).height == 0, "a delta row is outside the ucmix gate"
F = pl.read_parquet(inp)
Z = F.join(D, on=K, how="left", maintain_order="left")
v = pl.col(T).cast(pl.Float64) + (W / 0.5 - 1.0) * pl.col("d").fill_null(0.0)
out = Z.select(K, pl.when(pl.col("d").is_not_null()).then(v.clip(lower_bound=0).round(0)).otherwise(pl.col(T).cast(pl.Float64)).cast(pl.Int32).alias(T))
assert out[K].equals(F[K]) and out[T].null_count() == 0 and bool((out[T] >= 0).all())
out = out.cast(F.schema)
out.write_parquet(outp)
dd = (out[T].cast(pl.Float64) - F[T].cast(pl.Float64)).to_numpy()
print(f"claimed by rules B/C: {claimed.len()}; ucmix rows re-weighted: {D.height}; rows changed {int((dd != 0).sum())}; "
      f"sum d^2/670,790 {float((dd ** 2).sum()) / 670790:.1f} s^2; max|d| {int(np.abs(dd).max())}; wrote {outp}")
