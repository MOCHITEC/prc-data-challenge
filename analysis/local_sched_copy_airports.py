"""Schedule-copy labels (|y - (MVT - SCHED)| <= 12 s) by airport, lane and MVT-SCHED band, 2025 departures, and how the
OOF prediction (p = b + c_C, 7 months) fares on them (local).

    .venv/bin/python analysis/local_sched_copy_airports.py
"""
import glob
import os

import polars as pl

pl.Config.set_tbl_rows(300); pl.Config.set_tbl_width_chars(300)
S = os.environ["SCRATCH"]
O = S + "/agents/lirf/"


def main():
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "FLIGHT_mvt"]
    d = (pl.concat([pl.scan_parquet(f).select(cols) for f in sorted(glob.glob("data/raw/training_*.parquet"))])
         .filter(pl.col("PHASE_mvt") == "DEP")
         .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(), un=pl.col("AOBT_3_flt").is_null())
         .with_columns(copy=((pl.col("TAXITIME_SEC_mvt") - pl.col("ms")).abs() <= 12))
         .with_columns(band=(pl.col("ms") / 60).cut([-1, 5, 10, 20, 30, 60, 120, 240]))
         .collect())
    d.select("MVT_ID_mvt", "ADEP_mvt", "ms", "un", "copy", "FLIGHT_mvt", "TAXITIME_SEC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt").write_parquet(O + "dep_2025_copyflags.parquet")
    print("copy share by airport x lane")
    print(d.group_by("ADEP_mvt", "un").agg(n=pl.len(), copy=pl.col("copy").sum(), share=pl.col("copy").mean()).sort("ADEP_mvt", "un"))
    print("copies by band (minutes of MVT-SCHED) per airport x lane, airports with >= 50 copies")
    big = d.group_by("ADEP_mvt").agg(c=pl.col("copy").sum()).filter(pl.col("c") >= 50)["ADEP_mvt"]
    print(d.filter(pl.col("ADEP_mvt").is_in(big)).group_by("ADEP_mvt", "un", "band").agg(n=pl.len(), copy=pl.col("copy").sum(), share=pl.col("copy").mean().round(4)).sort("ADEP_mvt", "un", "band"))
    cv = pl.read_parquet(S + "/learn/cv_rows_months7_v2.parquet").with_columns(p=pl.col("b") + pl.col("c_C"))
    j = cv.join(d.select("MVT_ID_mvt", "ms", "un", "copy"), on="MVT_ID_mvt", how="inner")
    N = cv.height
    print("OOF on copies vs others (MSE contribution = SSE / N_oof)")
    print(j.group_by("ADEP_mvt", "copy").agg(n=pl.len(), mse_contrib=((pl.col("p") - pl.col("y")) ** 2).sum() / N,
                                              rmse=((pl.col("p") - pl.col("y")) ** 2).mean().sqrt(), bias=(pl.col("p") - pl.col("y")).mean(),
                                              sse_if_ms=((pl.col("ms") - pl.col("y")) ** 2).sum() / N).sort("ADEP_mvt", "copy"))


if __name__ == "__main__":
    main()
