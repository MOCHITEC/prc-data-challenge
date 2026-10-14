"""Apply the LIRF unmatched-lane label-artefact rules to a submission and print per-row changes (local; writes only to
$SCRATCH/agents/lirf/).

  C7r  C7 with mechanism-revised q: wrap needs the off-block to fall on a later date than SCHED and the re-dated block
       (off-block - 1 day) to land within about [SCHED - 10.5 h, SCHED + ~2 h]: TAY9LA-type rows (no midnight crossing)
       q = 0.05; rows whose re-dated block is > SCHED + 4 h (NOSOS431, +6.7 h) q = 0.15; otherwise C7 bands.
  C    7 h <= (MVT - SCHED - 1100) / 3600 < 13.5 h -> MVT - SCHED (2025: 34/34 schedule copies)
  B    0 <= MVT - SCHED < 1 h and FLIGHT prefix with >= 1 such 2025 row of the class y = UTC offset + taxi (off-block
       entered in local time): q (offset + 865) + (1 - q) p, q = tz / (n + 1) of the prefix in 2025 (BEL 13/14)

    .venv/bin/python analysis/local_lirf_artefacts_apply.py <in.parquet> <out.parquet>
"""
import glob
import os
import sys

import polars as pl

pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300)
DAY = 86400
T_WRAP = 1080
T_TZ = 865


def unm(df):
    off = 3600 + pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.dst_offset().dt.total_seconds()
    return (df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
            .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                          pre=pl.col("FLIGHT_mvt").str.slice(0, 3).fill_null("?"), off=off))


def main(src, dst):
    tr = unm(pl.concat([pl.scan_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))]).collect())
    small = tr.filter(pl.col("ms").is_between(0, 3599)).with_columns(
        tz=((pl.col("TAXITIME_SEC_mvt") - pl.col("off")).is_between(240, 2700)) & ((pl.col("TAXITIME_SEC_mvt") - pl.col("ms")).abs() > 12))
    qp = small.group_by("pre").agg(t=pl.col("tz").sum(), n=pl.len()).filter(pl.col("t") >= 1)
    sub = pl.read_parquet(src)
    R = unm(pl.read_parquet("data/raw/ranking.parquet")).join(sub.rename({"TAXITIME_SEC_mvt": "p"}), on="MVT_ID_mvt").join(qp, on="pre", how="left")
    loc = lambda c: c.dt.convert_time_zone("Europe/Rome").dt.date()
    ob = pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=1100)
    R = R.with_columns(dh=(pl.col("ms") - 1100) / 3600, cross=(loc(ob) - loc(pl.col("SCHED_TIME_UTC_mvt"))).dt.total_days(),
                       qtz=(pl.col("t") / (pl.col("n") + 1)).fill_null(0))
    R = R.with_columns(k=((pl.col("ms") - T_WRAP) / DAY).round(0).clip(1))
    R = R.with_columns(redated=pl.col("dh") - 24 * pl.col("k"))
    qw = (pl.when(pl.col("cross") < 1).then(0.05).when(pl.col("redated") > 4).then(0.15)
          .when(pl.col("ms") >= 56000).then(0.65).otherwise(0.29))
    new = (pl.when(pl.col("ms") >= 50400).then(qw * (pl.col("k") * DAY + T_WRAP) + (1 - qw) * pl.col("ms"))
           .when(pl.col("dh").is_between(7, 13.5)).then(pl.col("ms").cast(pl.Float64))
           .when(pl.col("ms").is_between(0, 3599)).then(pl.col("qtz") * (pl.col("off") + T_TZ) + (1 - pl.col("qtz")) * pl.col("p"))
           .otherwise(pl.col("p")))
    R = R.with_columns(new=new.round(0).cast(pl.Int32), rule=pl.when(pl.col("ms") >= 50400).then(pl.lit("C7r")).when(pl.col("dh").is_between(7, 13.5)).then(pl.lit("C"))
                       .when(pl.col("ms").is_between(0, 3599)).then(pl.lit("B")).otherwise(pl.lit("-")))
    G = R.filter(pl.col("new") != pl.col("p"))
    print(G.select("MVT_ID_mvt", "FLIGHT_mvt", "rule", "ms", "cross", "redated", "qtz", "p", "new").sort("rule", "ms"))
    print(G.group_by("rule").agg(n=pl.len(), mean_abs_change=(pl.col("new") - pl.col("p")).abs().mean()))
    out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
    assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    out.write_parquet(dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
